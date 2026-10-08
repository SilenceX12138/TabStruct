import math

import torch
import torch.nn as nn
import torch.nn.init as nn_init
from torch import Tensor

from .activation import get_activation


# ================================================================
# =                                                              =
# =                     Basic models                             =
# =                                                              =
# ================================================================
class MLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: str,
        hidden_layer_list: list,
        batch_normalization: bool = True,
        dropout_rate: float = 0,
    ) -> None:
        """MLP for classification or regression

        Args:
            output_dim (int): number of nodes for the output layer of the prediction net, 1 (regression) or 2 (classification)
            activation (str): activation function of the prediction net: 'relu', 'l_relu', 'sigmoid', 'tanh', or 'none'
            hidden_layer_list (list): number of nodes for each hidden layer for the prediction net, example: [200,200]
        """
        super().__init__()

        self.num_classes = output_dim
        self.act = get_activation(activation)
        full_layer_list = [input_dim, *hidden_layer_list]
        self.fn = nn.Sequential()
        for i in range(len(full_layer_list) - 1):
            self.fn.add_module("fn{}".format(i), nn.Linear(full_layer_list[i], full_layer_list[i + 1]))
            self.fn.add_module("act{}".format(i), self.act)
            # use BN after activation has better performance
            if batch_normalization:
                self.fn.add_module("bn{}".format(i), nn.BatchNorm1d(full_layer_list[i + 1]))
            if dropout_rate > 0:
                self.fn.add_module("dropout{}".format(i), nn.Dropout(dropout_rate))

        self.head = nn.Sequential()
        self.head.add_module("head", nn.Linear(full_layer_list[-1], output_dim))

        # when using cross-entropy loss in pytorch, we do not need to use softmax.
        # self.head.add_module('softmax', nn.Softmax(-1))

    def forward(self, x):
        x_emb = self.fn(x)
        x = self.head(x_emb)

        return x


# ================================================================
# =                                                              =
# =                     Transformers                             =
# =                                                              =
# ================================================================
class Tokenizer(nn.Module):
    """
    Tokenizer for tabular data that converts numerical and categorical features into token embeddings.

    This module creates learnable embeddings for both numerical and categorical features, similar to
    how transformers tokenize text. Each feature is converted into a fixed-dimensional token representation.
    A special [CLS] token is prepended to capture global information.

    Note: The returned token embeddings puts numerical tokens first, followed by categorical tokens by default.

    Args:
        dim_numerical (int): Number of numerical features in the input data
        cardinality_list (list): List of cardinalities for each categorical feature.
                                 Example: [3, 5] means 2 categorical features with 3 and 5 unique values
        dim_token (int): Dimension of the token embeddings (output dimension for each token)
        bias (bool): Whether to add learnable bias terms to the token embeddings
    """

    def __init__(
        self,
        dim_numerical,
        cardinality_list,
        dim_token,
        bias,
    ):
        super().__init__()

        # === Params for categorical features ===
        # Create position indices for slicing one-hot encoded categorical features
        # cat_start_pos_list: starting positions [0, card[0], card[0]+card[1], ...]
        # cat_end_pos_list: ending positions [card[0], card[0]+card[1], ...]
        self.cat_start_pos_list = torch.tensor([0] + cardinality_list[:-1]).cumsum(0) if cardinality_list else None
        self.cat_end_pos_list = torch.tensor(cardinality_list).cumsum(0) if cardinality_list else None

        # Weight matrix for categorical embeddings: shape (total_cardinality, dim_token)
        # Each categorical value gets its own embedding vector
        self.cat_weight = nn.Parameter(Tensor(sum(cardinality_list), dim_token)) if cardinality_list else None
        if self.cat_weight is not None:
            nn.init.kaiming_uniform_(self.cat_weight, a=math.sqrt(5))

        # === Params for numerical features ===
        # Weight matrix for numerical features: shape (dim_numerical + 1, dim_token)
        # +1 accounts for the [CLS] token prepended to the sequence
        self.num_weight = nn.Parameter(Tensor(dim_numerical + 1, dim_token))
        # Initialize using Kaiming uniform (He initialization) for better gradient flow
        nn_init.kaiming_uniform_(self.num_weight, a=math.sqrt(5))

        # === Params for all features ===
        # Optional bias term for each feature (not including [CLS] token)
        d_bias = dim_numerical + len(cardinality_list)
        self.bias = nn.Parameter(Tensor(d_bias, dim_token)) if bias else None
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    def forward(self, x_num, x_cat_onehot):
        """
        Convert numerical and categorical features into token embeddings.

        Args:
            x_num (Tensor): Numerical features, shape (batch_size, dim_numerical)
                           Can be None if only categorical features exist
            x_cat_onehot (Tensor): One-hot encoded categorical features,
                                  shape (batch_size, sum(cardinality_list))
                                  Can be None if only numerical features exist

        Returns:
            Tensor: Token embeddings, shape (batch_size, n_tokens, dim_token)
                   where n_tokens = 1 (CLS) + dim_numerical + len(cardinality_list)
        """
        # === Prepare numerical features with [CLS] token ===
        # Prepend a [CLS] token (value=1) to the numerical features
        x_num = self._prepare_numerical(x_num, x_cat_onehot)

        # === Tokenize numerical features ===
        # Element-wise multiplication: each feature value scales its corresponding weight vector
        # Shape: (batch_size, dim_numerical+1, dim_token)
        # This is NOT equivalent to nn.Linear because each feature has its own independent weight vector
        # rather than sharing weights across features
        x = self.num_weight[None] * x_num[:, :, None]

        # === Tokenize categorical features ===
        if x_cat_onehot is not None:
            x_cat_embedding_list = []
            # Process each categorical feature separately
            for start, end in zip(self.cat_start_pos_list, self.cat_end_pos_list):
                # Extract one-hot slice for this categorical feature and multiply with its weights
                # This performs a weighted sum of embeddings based on the one-hot encoding
                # Shape: (batch_size, 1, dim_token)
                x_cat_embedding = x_cat_onehot[:, start:end].unsqueeze(1) @ self.cat_weight[start:end].unsqueeze(0)
                x_cat_embedding_list.append(x_cat_embedding)
            # Concatenate along the token dimension
            x = torch.cat([x, *x_cat_embedding_list], dim=1)

        # === Add bias ===
        if self.bias is not None:
            # Add zero bias for [CLS] token to maintain its initialization
            # Then add learned bias to all other feature tokens
            bias = torch.cat([torch.zeros(1, self.bias.shape[1], device=self.bias.device), self.bias])
            x = x + bias[None]

        return x

    def _prepare_numerical(self, x_num, x_cat):
        """
        Prepend a [CLS] token to numerical features.

        The [CLS] token (similar to BERT) serves as an aggregation point for global information
        and is typically used for downstream prediction tasks.

        Args:
            x_num (Tensor or None): Numerical features, shape (batch_size, dim_numerical)
            x_cat (Tensor or None): Categorical features (used only to infer batch size if x_num is None)

        Returns:
            Tensor: Numerical features with prepended [CLS] token, shape (batch_size, dim_numerical+1)
                   The [CLS] token has a constant value of 1.0
        """
        # Determine batch size from whichever input is available
        x_ref = x_num if x_cat is None else x_cat

        # Create [CLS] token with value 1.0 for all samples in the batch
        cls_token = torch.ones(len(x_ref), 1, device=x_ref.device)
        x_num_list = [cls_token]
        if x_num is not None:
            x_num_list.append(x_num)
        x_num = torch.cat(x_num_list, dim=1)

        return x_num

    @property
    def n_tokens(self):
        """
        Calculate the total number of tokens produced by this tokenizer.

        Returns:
            int: Total number of tokens = 1 (CLS) + dim_numerical + number of categorical features
        """
        # Number of numerical tokens includes the [CLS] token
        n_num_tokens = self.num_weight.shape[0]
        # Each categorical feature produces one token
        n_cat_tokens = 0 if self.cat_start_pos_list is None else len(self.cat_start_pos_list)

        return n_num_tokens + n_cat_tokens


class Reconstructor(nn.Module):
    """
    Reconstructor for decoding token embeddings back into numerical and categorical features.

    This module performs the inverse operation of tokenization, converting learned token embeddings
    back into the original feature space. It's commonly used in autoencoder architectures for
    self-supervised learning or feature reconstruction tasks.

    Args:
        dim_numerical (int): Number of numerical features to reconstruct
        cardinality_list (list): List of cardinalities for each categorical feature.
                                 Example: [3, 5] means 2 categorical features with 3 and 5 unique values
        dim_token (int): Dimension of the input token embeddings
    """

    def __init__(self, dim_numerical, cardinality_list, dim_token):
        super(Reconstructor, self).__init__()

        # === Params for numerical features ===
        self.dim_numerical = dim_numerical
        # Weight matrix for numerical reconstruction: shape (dim_numerical, dim_token)
        # Each numerical feature has its own weight vector for decoding
        self.num_weight = nn.Parameter(Tensor(dim_numerical, dim_token))
        # Xavier initialization with adjusted gain for stable gradients
        nn.init.xavier_uniform_(self.num_weight, gain=1 / math.sqrt(2))

        # === Params for categorical features ===
        # Create separate linear layers for each categorical feature
        # Each layer maps from token embedding space to logits over categorical values
        self.cat_recon_net_list = nn.ModuleList()
        for d in cardinality_list:
            # Linear layer: dim_token -> d (cardinality of this categorical feature)
            recon = nn.Linear(dim_token, d)
            nn.init.xavier_uniform_(recon.weight, gain=1 / math.sqrt(2))
            self.cat_recon_net_list.append(recon)

    def forward(self, h):
        """
        Reconstruct numerical and categorical features from token embeddings.

        Args:
            h (Tensor): Token embeddings, shape (batch_size, n_tokens, dim_token)
                       Expected order: [CLS], numerical tokens, categorical tokens
                       Note: [CLS] token is typically excluded before passing to this module

        Returns:
            tuple: (recon_x_num, recon_x_cat)
                - recon_x_num (Tensor): Reconstructed numerical features,
                                       shape (batch_size, dim_numerical)
                - recon_x_cat (list of Tensors): Reconstructed categorical logits,
                                                 each element has shape (batch_size, cardinality_i)
                                                 where cardinality_i is the number of classes for feature i
        """
        # === Split numerical and categorical parts ===
        # Slice token embeddings based on feature types
        # First dim_numerical tokens correspond to numerical features (excluding [CLS])
        h_num = h[:, : self.dim_numerical]
        # Remaining tokens correspond to categorical features
        h_cat = h[:, self.dim_numerical :]

        # === Reconstruct numerical features ===
        # Element-wise multiply each token embedding with its weight vector, then sum across dim_token
        # This is the inverse of the tokenization operation: h_num * weight -> scalar value
        # Shape: (batch_size, dim_numerical, dim_token) * (1, dim_numerical, dim_token) -> sum -> (batch_size, dim_numerical)
        recon_x_num = (h_num * self.num_weight.unsqueeze(0)).sum(dim=-1)

        # === Reconstruct categorical features ===
        # For each categorical feature, apply its corresponding linear layer
        # This produces logits over the categorical classes (before softmax)
        recon_x_cat = []
        for i, recon_net in enumerate(self.cat_recon_net_list):
            # h_cat[:, i] has shape (batch_size, dim_token)
            # recon_net outputs (batch_size, cardinality_i) logits
            recon_x_cat.append(recon_net(h_cat[:, i]))
        recon_x_cat = torch.cat(recon_x_cat, dim=-1) if len(recon_x_cat) > 0 else None

        return recon_x_num, recon_x_cat


class Transformer(nn.Module):
    """
    Transformer encoder for tabular data using PyTorch's built-in APIs.

    Implements a stack of transformer layers using nn.TransformerEncoder, each consisting of
    multi-head self-attention followed by a position-wise feed-forward network (FFN).
    Supports both pre-normalization (norm before sublayer) and post-normalization
    (norm after sublayer) architectures.

    Architecture per layer:
    - Pre-norm:  x = x + Sublayer(LayerNorm(x))
    - Post-norm: x = LayerNorm(x + Sublayer(x))

    Args:
        n_layers (int): Number of transformer layers to stack
        d_token (int): Dimension of token embeddings
        n_heads (int): Number of attention heads in multi-head attention
        d_ffn_factor (int): Factor to determine FFN hidden dimension (d_hidden = d_token * d_ffn_factor)
        residual_dropout (float): Dropout rate applied to residual connections
        norm_first (bool): If True, use pre-norm architecture (norm_first=True);
                           otherwise post-norm (norm_first=False)
    """

    def __init__(
        self,
        n_layers: int,
        d_token: int,
        n_heads: int,
        d_ffn_factor: int,
        residual_dropout=0.0,
        norm_first=True,
    ):
        super().__init__()

        # === Configuration ===
        self.norm_first = norm_first
        self.d_token = d_token
        self.n_heads = n_heads

        # Calculate hidden dimension for feed-forward network
        d_hidden = int(d_token * d_ffn_factor)

        # === Build transformer encoder using PyTorch's built-in API ===
        # Create a single encoder layer configuration
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,  # Dimension of token embeddings
            nhead=n_heads,  # Number of attention heads
            dim_feedforward=d_hidden,  # Hidden dimension in FFN
            dropout=residual_dropout,  # Dropout on residual connections
            activation="relu",  # Activation function in FFN
            batch_first=True,  # Expect input as (batch, seq, feature)
            norm_first=norm_first,  # Pre-norm if True, post-norm if False
        )

        # Stack multiple encoder layers
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_token) if norm_first else None,  # Final normalization for pre-norm
        )

    def forward(self, x):
        """
        Forward pass through all transformer layers.

        Each layer processes the input through:
        1. Multi-head self-attention with residual connection
        2. Position-wise feed-forward network with residual connection

        Args:
            x (Tensor): Token embeddings, shape (batch_size, n_tokens, d_token)

        Returns:
            Tensor: Transformed representations, shape (batch_size, n_tokens, d_token)
        """
        # Pass through transformer encoder
        # PyTorch's TransformerEncoder expects (batch, seq, feature) due to batch_first=True
        x = self.encoder(x)

        return x
