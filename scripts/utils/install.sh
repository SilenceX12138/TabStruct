# Install uv (if uv is not installed)
curl -Ls https://astral.sh/uv/install.sh | bash

# Install 3rd party dependencies
uv pip install "mostlyai[local]"
uv pip install dgl -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html
uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.2+cu121.html
uv pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# Install the current package
uv pip install .