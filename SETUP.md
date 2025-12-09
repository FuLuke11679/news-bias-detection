# SETUP

Quick, reproducible setup instructions for the Clash Royale Emote Predictor.

1) Create a Python virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS / Linux (zsh/bash)
# On Windows (PowerShell): .\.venv\Scripts\Activate.ps1
```

2) Upgrade pip and install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3) (Optional) Install GPU-enabled PyTorch

If you have a CUDA-capable GPU, follow the instructions on https://pytorch.org to install a matching `torch` / `torchvision` wheel for your CUDA version before running heavy training.

4) Verify installation

```bash
python -c "import torch, torchvision; print('torch', torch.__version__, 'cuda_available=', torch.cuda.is_available())"
python -c "import cv2, numpy, PIL, sklearn; print('basic deps ok')"
```

If you encounter wheel / compatibility errors (common on newer Python like 3.13):

- Preferred: install a supported Python (3.10/3.11) and recreate the venv. See notes below.
- Alternative (install PyTorch explicitly): use the official PyTorch index to install wheels matching your OS/Python/CUDA. Example CPU-only install:

```bash
# CPU-only (recommended on macOS without CUDA)
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio --extra-index-url https://pypi.org/simple
```

For GPU-enabled installs, pick the command from https://pytorch.org (choose your OS, package manager, Python version and CUDA version) and run the generated command.

5) Run the project

- Train: `python -m src.train --train_dir data/train --test_dir data/test --checkpoint_dir models/checkpoints`
- Evaluate: `python -m src.eval --test_dir data/test --checkpoint_path models/checkpoints/best_resnet.pt`
- Demo (webcam): `python -m src.realtime_demo --checkpoint_path models/checkpoints/best_resnet.pt --emote_assets_dir data/emotes`

Troubleshooting
- If images are grayscale, the dataset loader converts them to RGB; if you see channel mismatch errors, check `src/data/fer_dataset.py`.
- If imports from `src` fail when running notebooks, run the notebook from the repository root or add the repo root to `sys.path`.

Notes
- The repository assumes Python 3.8+ but has been tested with Python 3.10+. Adjust virtualenv Python as needed.

Python version tips
- If you use Python 3.13 and pip fails to find compatible wheels for `torch`/`torchvision`, prefer creating a venv with Python 3.10 or 3.11. This avoids many wheel compatibility issues.
- Tools: `pyenv` (Homebrew) or `conda` make it easy to install and switch Python versions.

