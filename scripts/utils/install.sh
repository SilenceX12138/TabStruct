# Install uv (if uv is not installed)
# uv >0.10.2 introduces a breaking change that causes the installation to fail, so we specify the version explicitly.
pip install uv==0.10.2

# Install the current package
uv pip install .

# Install 3rd party dependencies
uv pip install "mostlyai[local]"
uv pip install dgl==2.4.0+cu121 -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html
uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.2+cu121.html
uv pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
