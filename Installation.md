# Installation Guide

Before you start, you can optionally configure a pip source for faster downloads (especially for users in China):
```bash
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
pip config set install.trusted-host mirrors.aliyun.com
```

## Install Inferix

### Option 1: Install with pip (Recommended)

**This is the recommended approach for most users as it automatically handles PYTHONPATH.**

```bash
# Install uv
pip install uv

# Create a virtual environment (this will create a .venv folder at current project)
uv venv -p python3.11
source .venv/bin/activate

# Install Inferix in editable mode
pip install -e .

# Install flash-attn (this may take a long time and may need to retry several times when installation fails)
pip install flash-attn --no-build-isolation
```

### Option 2: Manual Installation (Develop Mode)

```bash
# Install uv
pip install uv

# Create a virtual environment (this will create a .venv folder at current project)
uv venv -p python3.11
source .venv/bin/activate

# Make sure pip is installed in the virtual environment
uv pip install pip --python ".venv/bin/python"

# Build basic torch related packages
pip install -r requirements-torch.txt

# Build Inferix related packages
pip install -r requirements.txt

# Set PYTHONPATH
export PYTHONPATH=`pwd`:$PYTHONPATH

# Install flash-attn (this may take a long time and may need to retry several times when installation fails)
pip install flash-attn --no-build-isolation
```

## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (recommended: A100, H100, or equivalent)
- **VRAM**: Minimum 24GB for basic inference (40GB+ recommended for larger models)
- **RAM**: Minimum 32GB system memory
- **Storage**: Minimum 50GB free disk space for model weights and temporary files

### Software
- **OS**: Linux (Ubuntu 20.04+ recommended) or macOS (12.0+)
- **Python**: 3.10 or 3.11
- **CUDA**: 12.1 or higher
- **Git**: For downloading model weights

## Next Steps

After successful installation, you can:
1. Download model weights following the instructions in each model's README
2. Run example scripts in the [example/](example/) directory
3. Refer to [model_integration_guide.md](model_integration_guide.md) for integrating your own models