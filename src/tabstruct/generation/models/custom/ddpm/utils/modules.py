import math

import torch
import torch.nn as nn
from einops import rearrange

from src.tabstruct.common.model.utils.component import MLP, Reconstructor, Tokenizer, Transformer


class UniModVAE(nn.Module):
    """Unified Multimodal Variational Autoencoder for tabular data.

    This VAE handles mixed-type tabular data (numerical + categorical) by:
    1. Tokenizing features into a unified representation
    2. Encoding to a probabilistic latent space (mean + log-variance)
    3. Sampling latent codes via reparameterization trick
    4. Decoding back to original feature space

    Architecture:
    - Tokenizer: Converts mixed-type data to token embeddings
    - Encoder (dual): Separate transformers for mean and log-variance
    - Decoder: Transformer for latent-to-feature reconstruction
    - Detokenizer: Converts tokens back to numerical/categorical predictions

    Args:
        d_numerical: Number of numerical features in the dataset.
        categories: List of cardinalities for categorical features (e.g., [5, 10, 3]).
        num_layers: Number of transformer layers in encoder/decoder.
        d_token: Token embedding dimension (default: 4).
        n_head: Number of attention heads in transformers (default: 1).
        factor: Feed-forward expansion factor in transformers (default: 32).
        bias: Whether to use bias terms in tokenizer (default: True).

    Attributes:
        latent_dim: Dimension of the latent space (d_token * num_features).
    """

    def __init__(
        self,
        d_numerical: int,
        categories: list[int],
        num_layers: int,
        d_token: int = 4,
        n_head: int = 1,
        factor: int = 32,
        bias: bool = True,
    ) -> None:
        super().__init__()

        # Store feature dimensions
        self.d_numerical: int = d_numerical
        self.categories: list[int] = categories
        self.num_categorical: int = len(categories)
        self.num_features: int = d_numerical + self.num_categorical

        # Latent space dimension: flattened token representation
        self.latent_dim: int = d_token * self.num_features

        # Tokenizer: Convert mixed-type tabular data to unified token embeddings
        self.tokenizer = Tokenizer(d_numerical, categories, d_token, bias=bias)

        # Dual encoder: Separate networks for mean and log-variance of latent distribution
        # This allows the model to learn both the central tendency and uncertainty
        self.encoder_mu = Transformer(num_layers, d_token, n_head, factor)
        self.encoder_log_var = Transformer(num_layers, d_token, n_head, factor)

        # Decoder: Reconstruct features from latent codes
        self.decoder = Transformer(num_layers, d_token, n_head, factor)

        # Detokenizer: Convert token embeddings back to original feature space
        self.detokenizer = Reconstructor(d_numerical, categories, d_token)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> dict[str, torch.Tensor]:
        """Full VAE forward pass: encode → sample → decode.

        Args:
            x_num: Numerical features [batch_size, d_numerical].
            x_cat: Categorical features [batch_size, num_categorical].

        Returns:
            Dictionary containing:
            - x_num_recon: Reconstructed numerical features [batch_size, d_numerical]
            - x_cat_recon: Reconstructed categorical logits [batch_size, sum(categories)]
            - z_mu: Latent mean [batch_size, latent_dim]
            - z_log_var: Latent log-variance [batch_size, latent_dim]
        """
        # Encode input to probabilistic latent space
        encode_dict = self.encode(x_num, x_cat)
        latent_sample = encode_dict["z"]
        latent_mean = encode_dict["z_mu"]
        latent_log_variance = encode_dict["z_log_var"]

        # Decode latent sample back to feature space
        decode_dict = self.decode(latent_sample)
        x_num_reconstructed = decode_dict["x_num_recon"]
        x_cat_reconstructed = decode_dict["x_cat_recon"]

        return {
            "x_num_recon": x_num_reconstructed,
            "x_cat_recon": x_cat_reconstructed,
            "z_mu": latent_mean,
            "z_log_var": latent_log_variance,
        }

    def encode(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> dict[str, torch.Tensor]:
        """Encode input features to probabilistic latent representation.

        Args:
            x_num: Numerical features [batch_size, d_numerical].
            x_cat: Categorical features [batch_size, num_categorical].

        Returns:
            Dictionary containing:
            - z: Sampled latent code [batch_size, latent_dim]
            - z_mu: Latent mean [batch_size, latent_dim]
            - z_log_var: Latent log-variance [batch_size, latent_dim]
        """
        # Step 1: Tokenize mixed-type input to unified representation
        # Output shape: [batch_size, num_features + 1, d_token] (includes CLS token)
        token_embeddings = self.tokenizer(x_num, x_cat)

        # Step 2: Prepare encoder input (exclude CLS token at position 0)
        # CLS token is used for aggregation but not encoded
        encoder_input = token_embeddings[:, 1:, :]  # [batch_size, num_features, d_token]

        # Step 3: Encode to latent mean via transformer
        # Shape: [batch_size, num_features, d_token]
        latent_mean_tokens = self.encoder_mu(encoder_input)
        # Flatten tokens: [batch_size, num_features, d_token] -> [batch_size, latent_dim]
        latent_mean = rearrange(latent_mean_tokens, "batch features tokens -> batch (features tokens)")

        # Step 4: Encode to latent log-variance via separate transformer
        # Using log-variance ensures positive variance via exp()
        # Shape: [batch_size, num_features, d_token]
        latent_log_var_tokens = self.encoder_log_var(encoder_input)
        # Flatten tokens: [batch_size, num_features, d_token] -> [batch_size, latent_dim]
        latent_log_variance = rearrange(latent_log_var_tokens, "batch features tokens -> batch (features tokens)")

        # Step 5: Sample from latent distribution using reparameterization trick
        # z = μ + σ * ε, where ε ~ N(0, 1)
        latent_sample = self._reparameterize(latent_mean, latent_log_variance)

        return {
            "z": latent_sample,
            "z_mu": latent_mean,
            "z_log_var": latent_log_variance,
        }

    def decode(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Decode latent codes back to feature space.

        Args:
            z: Latent codes [batch_size, latent_dim].

        Returns:
            Dictionary containing:
            - x_num_recon: Reconstructed numerical features [batch_size, d_numerical]
            - x_cat_recon: Reconstructed categorical logits [batch_size, sum(categories)]
        """
        # Step 1: Reshape flattened latent to token format
        # [batch_size, latent_dim] -> [batch_size, num_features, d_token]
        batch_size = z.shape[0]
        decoder_input = z.reshape(batch_size, self.num_features, -1)

        # Step 2: Process through decoder transformer
        # Refines latent representation for better reconstruction
        decoded_tokens = self.decoder(decoder_input)

        # Step 3: Convert tokens back to numerical and categorical predictions
        x_num_reconstructed, x_cat_reconstructed = self.detokenizer(decoded_tokens)

        return {
            "x_num_recon": x_num_reconstructed,
            "x_cat_recon": x_cat_reconstructed,
        }

    def _reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: sample from N(μ, σ²) via N(0, 1).

        This allows backpropagation through the sampling operation by expressing
        the random sample as: z = μ + σ * ε, where ε ~ N(0, 1).

        Args:
            mu: Latent mean [batch_size, latent_dim].
            log_var: Latent log-variance [batch_size, latent_dim].

        Returns:
            Sampled latent codes [batch_size, latent_dim].
        """
        # Convert log-variance to standard deviation: σ = exp(0.5 * log(σ²))
        std = torch.exp(0.5 * log_var)

        # Sample noise from standard normal distribution
        epsilon = torch.randn_like(std)

        # Apply reparameterization: z = μ + σ * ε
        return mu + epsilon * std


class UniModMLP(nn.Module):
    """
    Unified Multi-Modal MLP for tabular diffusion denoising.

    This class implements a transformer-based denoising network that handles both numerical
    and categorical features in a unified manner. It uses a tokenization approach where
    features are converted to tokens, processed through transformer layers, and then
    reconstructed back to the original feature space.

    Architecture:
    1. Tokenizer: Converts mixed-type tabular data to token embeddings
    2. Encoder: Transformer layers for feature interaction modeling
    3. MLP: Timestep-conditioned diffusion backbone
    4. Decoder: Transformer layers for refined feature reconstruction
    5. Detokenizer: Converts tokens back to numerical/categorical predictions

    Args:
        d_numerical: Number of numerical features
        categories: List of category counts for categorical features
        num_layers: Number of transformer layers for encoder/decoder
        d_token: Token embedding dimension
        n_head: Number of attention heads in transformers (default: 1)
        factor: Feed-forward expansion factor in transformers (default: 4)
        bias: Whether to use bias in tokenizer (default: True)
        dim_t: Hidden dimension for the MLP diffusion component (default: 512)

    Input:
        x_num: Numerical features [batch_size, d_numerical]
        x_cat: Categorical features [batch_size, len(categories)]
        timesteps: Diffusion timesteps [batch_size]

    Output:
        x_num_pred: Predicted numerical features [batch_size, d_numerical]
        x_cat_pred: Predicted categorical logits [batch_size, sum(categories)] (UNNORMALIZED)
    """

    def __init__(
        self,
        d_numerical,
        categories,
        num_layers,
        d_token=4,
        n_head=1,
        factor=32,
        bias=True,
        dim_t=1024,
    ):
        super().__init__()
        # Store feature dimensions for later use
        self.d_numerical = d_numerical
        self.categories = categories

        # Tokenizer: Convert tabular data (numerical + categorical) to token embeddings
        # This creates a unified token representation for mixed-type data
        self.tokenizer = Tokenizer(d_numerical, categories, d_token, bias=bias)

        # Encoder: Transformer layers for modeling feature interactions
        # Processes tokenized features to capture complex dependencies
        self.encoder = Transformer(num_layers, d_token, n_head, factor)

        # MLP: Timestep-conditioned diffusion backbone in token space
        # Calculate input dimension for the MLP (flattened token representation)
        d_in = d_token * (d_numerical + len(categories))
        # Core denoising component with temporal conditioning
        self.mlp = MLPDiffusion(input_dim=d_in, hidden_dim=dim_t)

        # Decoder: Transformer layers for refined reconstruction
        # Further processes the MLP output to improve feature quality
        self.decoder = Transformer(num_layers, d_token, n_head, factor)

        # Detokenizer: Convert tokens back to numerical/categorical predictions
        # Reconstructs the original feature space from processed tokens
        self.detokenizer = Reconstructor(d_numerical, categories, d_token)

    def forward(self, x_num, x_cat, timesteps):
        """
        Forward pass through the unified multi-modal denoising network.

        Args:
            x_num: Numerical features [batch_size, d_numerical]
            x_cat: Categorical features [batch_size, len(categories)]
            timesteps: Diffusion timesteps [batch_size]

        Returns:
            Tuple of (numerical_predictions, categorical_logits):
            - x_num_pred: Denoised numerical features [batch_size, d_numerical]
            - x_cat_pred: Categorical logits [batch_size, sum(categories)] (unnormalized)
        """
        # Step 1: Tokenize mixed-type input data to unified token representation
        # Creates token embeddings with CLS token for global context
        e = self.tokenizer(x_num, x_cat)

        # Step 2: Prepare input for encoder (skip CLS token)
        # The CLS token is used for aggregation but not processed through encoder
        encoder_input = e[:, 1:, :]  # ignore the first CLS token.

        # Step 3: Process tokens through transformer encoder
        # Models complex feature interactions and dependencies
        y = self.encoder(encoder_input)

        # Step 4: Flatten and apply timestep-conditioned MLP
        # Core diffusion denoising with temporal information
        pred_y = self.mlp(y.reshape(y.shape[0], -1), timesteps)

        # Step 5: Reshape and decode through transformer decoder
        # Refines the MLP predictions using additional transformer layers
        pred_e = self.decoder(pred_y.reshape(*y.shape))

        # Step 6: Detokenize back to original feature space
        # Converts processed tokens to numerical/categorical predictions
        x_num_pred, x_cat_pred = self.detokenizer(pred_e)

        return x_num_pred, x_cat_pred


class MLPDiffusion(nn.Module):
    """
    Multi-Layer Perceptron with timestep conditioning for diffusion models.

    This class implements a simple yet effective denoising network architecture that combines
    a linear projection, timestep embedding, and an MLP backbone. The timestep conditioning
    is applied via additive embedding (sum mode) to incorporate temporal information into
    the denoising process.

    Architecture:
    1. Linear projection to expand input to hidden dimension
    2. Timestep embedding added to the projected features
    3. Multi-layer perceptron for feature transformation

    Args:
        input_dim: Dimensionality of input features
        hidden_dim: Hidden dimension for the network (default: 512)
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
    ):
        super().__init__()

        # Linear projection to map input to hidden dimension
        # This expands the feature space before timestep conditioning
        self.proj = nn.Linear(
            input_dim,
            hidden_dim,
        )

        # Timestep embedding module for temporal conditioning
        # Uses additive (sum) mode to integrate time information
        self.timestep_embed = TimestepEmbedding(
            input_dim=hidden_dim,
            encode_mode="sum",
            encode_reduce=None,
        )

        # Main MLP backbone for feature transformation
        # Uses SiLU activation and progressively reduces dimensions
        self.mlp = MLP(
            input_dim=self.timestep_embed.output_dim,
            output_dim=input_dim,
            activation="silu",
            hidden_layer_list=[hidden_dim * 2, hidden_dim * 2, hidden_dim],
            batch_normalization=False,
            dropout_rate=0.0,
        )

    def forward(self, x, timesteps):
        """
        Forward pass through the MLP diffusion network.

        Args:
            x: Input tensor [batch_size, input_dim] - flattened feature representation
            timesteps: Timestep tensor [batch_size] - current diffusion timestep

        Returns:
            Denoised output tensor [batch_size, input_dim] with same shape as input
        """
        # Project input to hidden dimension first
        # This allows for better integration with timestep embeddings
        x = self.proj(x)

        # Add timestep conditioning via embedding summation
        # This injects temporal information into the feature representation
        x = self.timestep_embed(x, timesteps)

        # Apply main MLP transformation for denoising
        # The network learns to predict noise or clean data based on conditioning
        x = self.mlp(x)

        return x


class TimestepEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding module for diffusion models.

    This class creates positional encodings for timesteps using sinusoidal functions,
    similar to the positional encodings used in Transformers. It supports both discrete
    and continuous timesteps, with flexible encoding modes for integration with features.

    The embeddings use alternating sine and cosine functions at different frequencies:
    - Even indices: sin(timestep / 10000^(2i/d))
    - Odd indices: cos(timestep / 10000^(2i/d))

    Args:
        input_dim: Dimension of the input features and output embeddings
        timestep_max: Maximum timestep value for pre-computing discrete embeddings (default: 10000)
        encode_mode: How to combine embeddings with input features:
                    - "sum": Add embeddings to input (default)
                    - "concat": Concatenate embeddings with input
                    - None: Return input unchanged
        encode_reduce: Reduction method for concatenation mode:
                      - "mean": Average the embeddings before concatenation
                      - "sum": Sum the embeddings before concatenation
                      - None: Use full embeddings (default)
    """

    def __init__(
        self,
        input_dim: int,
        timestep_max: int = 10000,
        encode_mode: str | None = "sum",
        encode_reduce: str | None = None,
    ):
        super().__init__()
        # === Sanity check ===
        # Ensure encode_reduce is only used with concat mode
        if encode_mode in ["sum", None]:
            if encode_reduce is not None:
                raise ValueError("encode_reduce must be None when encode_mode is 'sum'")

        # === Save the parameters ===
        self.input_dim = input_dim
        self.encode_mode = encode_mode
        self.encode_reduce = encode_reduce

        # === Create the positional encoding matrix ===
        # Pre-compute embeddings for discrete timesteps (efficiency)
        frequency_terms_discrete = self._compute_frequency_discrete(timestep_max)
        # Pre-compute frequency terms for continuous timesteps (flexibility)
        frequency_terms_continuous = self._compute_frequency_continuous(timestep_max)
        self.register_buffer("frequency_terms_discrete", frequency_terms_discrete)
        self.register_buffer("frequency_terms_continuous", frequency_terms_continuous)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Apply timestep embeddings to input features.

        Args:
            x: Input feature tensor [batch_size, input_dim]
            timesteps: Timestep tensor [batch_size] - can be discrete (int) or continuous (float)

        Returns:
            Feature tensor with timestep information incorporated based on encode_mode
        """
        # Generate timestep embeddings for the current batch
        timestep_embeddings_batch = self._create_timestep_embeddings(timesteps)

        # Apply the specified encoding mode
        match self.encode_mode:
            case "sum":
                # Additive integration: preserves input dimensionality
                x = x + timestep_embeddings_batch
            case "concat":
                # Concatenative integration: may increase dimensionality
                if self.encode_reduce == "mean":
                    # Reduce embeddings to single dimension before concatenation
                    timestep_embeddings_batch = timestep_embeddings_batch.mean(dim=1, keepdim=True)
                elif self.encode_reduce == "sum":
                    # Reduce embeddings to single dimension before concatenation
                    timestep_embeddings_batch = timestep_embeddings_batch.sum(dim=1, keepdim=True)
                # Concatenate along feature dimension
                x = torch.cat([x, timestep_embeddings_batch], dim=-1)
            case None:
                # No timestep conditioning: return input unchanged
                x = x
            case _:
                raise ValueError(f"Unknown encode_mode: {self.encode_mode}")

        return x

    # ================================================================
    # =                                                              =
    # =                         Utils                                =
    # =                                                              =
    # ================================================================
    def _compute_frequency_discrete(self, timestep_max: int) -> torch.Tensor:
        """
        Pre-compute sinusoidal embeddings for all discrete timesteps up to timestep_max.

        This creates a lookup table for efficient embedding retrieval when using
        discrete (integer) timesteps.

        Args:
            timestep_max: Maximum timestep value to pre-compute

        Returns:
            Tensor of shape [timestep_max, input_dim] containing pre-computed embeddings
        """
        # position: (max_len, 1) - timestep indices from 0 to timestep_max-1
        position = torch.arange(timestep_max).unsqueeze(1)
        # div_term: (d_model//2,) - frequency scaling factors for each dimension pair
        half_dim = self.input_dim // 2
        div_term = torch.exp(torch.arange(half_dim) * (-math.log(timestep_max) / self.input_dim))
        # timestep_embeddings: (max_len, d_model) - output embedding matrix
        timestep_embeddings = torch.zeros(timestep_max, self.input_dim)

        # === Fill the positional encoding matrix ===
        # Apply sin to even indices in the array; 2i
        # Apply cos to odd indices in the array; 2i+1
        if self.input_dim % 2 == 0:
            # For even input_dim, all positions are properly paired
            timestep_embeddings[:, 0::2] = torch.sin(position * div_term)
            timestep_embeddings[:, 1::2] = torch.cos(position * div_term)
        else:
            # For odd input_dim, handle the last unpaired dimension
            timestep_embeddings[:, 0:-1:2] = torch.sin(position * div_term)
            timestep_embeddings[:, 1::2] = torch.cos(position * div_term)
            # Fill the last dimension with sine of the first frequency
            timestep_embeddings[:, -1] = torch.sin(position[:, 0] * div_term[0])

        return timestep_embeddings

    def _compute_frequency_continuous(self, timestep_max: int) -> torch.Tensor:
        """
        Create frequency terms for continuous timestep embeddings.

        This method pre-computes the frequency components that will be used
        to generate sinusoidal embeddings for any continuous timestep values.

        Args:
            timestep_max: Maximum timestep value, used as the base for frequency scaling

        Returns:
            Tensor of shape [input_dim // 2] containing frequency terms
        """
        # Number of frequency components (half the embedding dimension)
        half_dim = self.input_dim // 2

        # Create frequency indices: [0, 1, 2, ..., half_dim-1]
        freq_indices = torch.arange(start=0, end=half_dim, dtype=torch.float32)

        # Normalize frequency indices to [0, 1) range
        normalization_factor = half_dim
        normalized_freqs = freq_indices / normalization_factor

        # Convert to exponentially decreasing frequencies
        # Higher indices get lower frequencies for better positional resolution
        frequency_terms = (1.0 / timestep_max) ** normalized_freqs

        return frequency_terms

    def _create_timestep_embeddings(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Generate sinusoidal embeddings for given timesteps using pre-computed frequency terms.

        This method automatically detects whether timesteps are discrete or continuous
        and applies the appropriate embedding strategy.

        Args:
            timesteps: Tensor of shape [batch_size] containing timestep values

        Returns:
            Tensor of shape [batch_size, input_dim] containing the sinusoidal embeddings
        """
        # Check if timesteps are continuous (float) or discrete (int)
        if not timesteps.dtype.is_floating_point:
            # Use discrete embedding (lookup from pre-computed table)
            # This is more efficient for integer timesteps
            return self.frequency_terms_discrete[timesteps]
        else:
            # Use continuous embedding (compute on-the-fly)
            # Generate sinusoidal embeddings for arbitrary float timesteps
            embedding = torch.outer(timesteps, self.frequency_terms_continuous)
            timestep_embeddings = torch.zeros((timesteps.shape[0], embedding.shape[1] * 2), device=embedding.device)
            # Interleave sine and cosine values
            timestep_embeddings[:, 0::2] = embedding.sin()
            timestep_embeddings[:, 1::2] = embedding.cos()

            if self.input_dim % 2 == 1:
                # If input_dim is odd, we need to add an extra dimension
                # We duplicate the first sine value to maintain the target dimension
                timestep_embeddings = torch.cat([timestep_embeddings, timestep_embeddings[:, :1].clone()], dim=1)

            return timestep_embeddings

    # ================================================================
    # =                                                              =
    # =                      Properties                              =
    # =                                                              =
    # ================================================================
    @property
    def output_dim(self) -> int:
        """
        Calculate and cache the output dimension based on encoding configuration.

        The output dimension depends on the encode_mode:
        - "sum" or None: Same as input_dim (additive integration)
        - "concat" with reduction: input_dim + 1 (concatenate single reduced value)
        - "concat" without reduction: input_dim * 2 (concatenate full embedding)

        Returns:
            Integer representing the output feature dimension
        """
        # Use cached value if already computed
        if hasattr(self, "_output_dim"):
            return getattr(self, "_output_dim")

        # Compute output dimension based on encoding mode
        match self.encode_mode:
            case "sum" | None:
                # Additive mode preserves input dimensionality
                output_dim = self.input_dim
            case "concat":
                # Concatenation mode may change dimensionality
                if self.encode_reduce in ["mean", "sum"]:
                    # When reducing timestep embeddings, we add 1 dimension
                    output_dim = self.input_dim + 1
                else:
                    # Otherwise, we concatenate full embedding (doubling the size)
                    output_dim = self.input_dim * 2
            case _:
                raise ValueError(f"Unknown encode_mode: {self.encode_mode}")

        # Cache the computed dimension for future calls
        self._output_dim = output_dim

        return self._output_dim
