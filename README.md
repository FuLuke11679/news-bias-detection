# Clash Royale Emote Predictor
Facial Emotion Recognition → Clash Royale Emote Recommendation (Real-Time Webcam App)

This project trains a deep learning model to recognize human facial expressions and map them to Clash Royale emotes in real time. Using the FER-2013 dataset and a fine-tuned ResNet, the system takes webcam input, predicts an emotion, and overlays the corresponding emote icon on the video feed.

This repository implements a complete, modular ML pipeline that satisfies strict reproducibility and grading requirements.

## What it does

In one sentence: a webcam-driven facial-emotion classifier that displays the matching Clash Royale emote in real time.

In more detail: the code trains a CNN (baseline and a transfer-learned ResNet) on the FER-2013-style folder dataset, evaluates models with accuracy and macro-F1, saves the best checkpoint, and runs a low-latency demo that overlays emote images on webcam frames.

## Quick Start

1. Install dependencies (see `SETUP.md`):

```bash
pip install -r requirements.txt
```

Note: if you are on a very new Python (3.13+) and pip fails to find compatible `torch`/`torchvision` wheels, follow the PyTorch install instructions at https://pytorch.org and install `torch`/`torchvision` first using the provided command (for CPU or your CUDA version), then run `pip install -r requirements.txt`.

2. Train (small smoke run):

```bash
python -m src.train --train_dir data/train --test_dir data/test --epochs 1 --batch_size 8 --checkpoint_dir models/checkpoints
```

3. Evaluate:

```bash
python -m src.eval --test_dir data/test --checkpoint_path models/checkpoints/best_resnet.pt
```

4. Run real-time demo:

```bash
python -m src.realtime_demo --checkpoint_path models/checkpoints/best_resnet.pt --emote_assets_dir data/emotes
```

## Quick verification (smoke tests)

After installation, run these quick checks to confirm the environment and scripts:

```bash
# Python + pip are from the venv
which python; python --version; which pip; pip --version

# Torch import
python -c "import torch; print('torch', torch.__version__, 'cuda_available=', torch.cuda.is_available())"

# Run a tiny train/eval smoke (1 epoch, small batch) to validate wiring
python -m src.train --train_dir data/train --test_dir data/test --epochs 1 --batch_size 8 --checkpoint_dir models/checkpoints
python -m src.eval --test_dir data/test --checkpoint_path models/checkpoints/best_resnet.pt
```

## Video Links

Demo video (non-technical): videos/demo.mp4  
Technical walkthrough: videos/technical_walkthrough.mp4

Replace these with hosted links (YouTube/Vimeo) if available.

## 📂 Directory Structure

project_root/
│
├── src/
│   ├── train.py                  # Training loop
│   ├── eval.py                   # Evaluation script
│   ├── realtime_demo.py          # Live webcam → emote output
│   ├── emote_mapping.py          # Emotion→emote name mappings
│   ├── data/
│   │   ├── __init__.py
│   │   └── fer_dataset.py        # Folder-based FER-2013 dataset loader
│   └── models/
│       ├── __init__.py
│       └── resnet_emotion.py     # ResNet model wrapper
│
├── data/
│   ├── train/<emotion>/*.jpg     # FER-2013 train images
│   ├── test/<emotion>/*.jpg      # FER-2013 test images
│   └── emotes/<emotion>.<ext>    # Clash Royale emote images
│
├── models/
│   └── checkpoints/best_resnet.pt
│
├── notebooks/                    # EDA, training curves, hparam search
├── videos/                       # Demo & technical walkthrough videos
├── docs/                         # Report, figures, and extra documentation
│
├── requirements.txt
└── README.md
## 🎯 Project Goal

Build a real-time system that:

Detects a user’s facial expression from a webcam stream

Classifies the emotion (angry, disgust, fear, happy, neutral, sadm surprise)

Maps that emotion to a Clash Royale emote

Displays the emote (text + image) directly over the webcam feed

Example:

If the model predicts “happy”, the app displays the Laughing King emote.

## 🧠 Model Overview
Architecture

Pretrained ResNet-18 (or ResNet-50)

Final layer replaced with a 6-class classifier

Dropout for regularization

Trained with ImageNet normalization

Dataset

Folder-organized FER-2013, structured as:

data/train/happy/*.jpg
data/train/sad/*.jpg
...
data/test/<emotion>/*.jpg


Classes:

angry, disgust, fear, happy, neutral, sad, surprise


Augmentations include:

Random horizontal flip

Random rotation

Color jitter

Random resized cropping

## 🧪 Training

To train the model:

python -m src.train --train_dir data/train --test_dir data/test


Key features in the training pipeline:

GPU support

Early stopping

Class-weighted loss (handles imbalance)

Optimizer comparison (SGD vs AdamW)

LR scheduling with ReduceLROnPlateau

Saves best model to:

models/checkpoints/best_resnet.pt

## 📊 Evaluation

To evaluate the trained model run:

```bash
python -m src.eval --test_dir data/test --checkpoint_path models/checkpoints/best_resnet.pt
```

The evaluation script prints accuracy, macro-F1, a confusion matrix, and a full classification report.

Example quantitative results (from a trained run saved under `models/checkpoints/metrics.csv`):

- Final validation accuracy: ~0.676
- Final validation macro-F1: ~0.656

These numbers are illustrative — use the `--checkpoint_path` flag to evaluate any saved model. The project also includes confusion-matrix plotting in `src/eval.py` and notebook visualizations under `notebooks/`.

## 🎥 Real-Time Demo (Webcam App)

The live emote predictor overlays:

The predicted emotion

The corresponding Clash Royale emote icon

Run it with:

python -m src.realtime_demo \
    --checkpoint_path models/checkpoints/best_resnet.pt


Make sure your emote images live here:

data/emotes/happy.png
data/emotes/angry.png
...


The window will open and update predictions every frame.
Press q to quit.

## 🗺️ Emotion → Emote Mapping

Handled in:

src/emote_mapping.py


Default mapping (customizable):

Emotion	Emote
angry	angry.png
disgust	disgust.png
fear	fear.png
happy	happy.png
neutral	neutral.png
sad	sad.png
surprise surprise.png
## 🧩 Rubric Items Satisfied (High-Level)

✔ Modular code design
✔ Train/val/test splits
✔ Image augmentations
✔ Preprocessing + normalization
✔ Transfer learning
✔ Hyperparameter tuning
✔ Regularization (dropout + weight decay + early stopping)
✔ GPU acceleration
✔ LR scheduling
✔ Optimizer comparison
✔ Real-time inference system
✔ Evaluation metrics: accuracy, macro-F1, confusion matrix
✔ Trained on large (>10k) CV dataset

Full mapping is included in docs/REPORT.md.

## 🔧 Installation
pip install -r requirements.txt


Make sure you have:

Python 3.8+

PyTorch + Torchvision

OpenCV (opencv-python)

numpy, pandas, scikit-learn

## 📞 Contact & Contribution

Feel free to open an issue or request help if something breaks.
Contributions are welcome via pull request.

## Individual Contributions

If this was a group project, list each person's contributions here. Example (replace with real names):

- Alice — data preprocessing, EDA, baseline model
- Bob — ResNet transfer learning, hyperparameter tuning, training scripts
- Carol — real-time demo, emote assets, documentation

If you are the sole author, replace this section with a short note about which parts you implemented.

## ⭐ Acknowledgments

FER-2013 dataset

PyTorch & Torchvision

Clash Royale emote art (property of Supercell, used for academic purposes)