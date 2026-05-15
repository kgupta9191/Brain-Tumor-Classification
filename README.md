# Brain Tumor Classification

Brain Tumor Classification is a PyTorch-based deep learning project for classifying brain MRI images into tumor-related classes using a fine-tuned **ResNet18** model.

The repository includes:
- A complete training and evaluation pipeline (`src/cnn.py`)
- Data ingestion from folder-structured image datasets
- Augmentation and preprocessing transforms
- Unit tests for core training/data components
- GitHub Actions CI for automated test runs

---

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [How the Pipeline Works](#how-the-pipeline-works)
- [Dataset Layout](#dataset-layout)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Run Training](#run-training)
- [Testing](#testing)
- [CI/CD](#cicd)
- [Configuration](#configuration)
- [Output Artifacts](#output-artifacts)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

This project trains a convolutional neural network to classify MRI scans.
It uses transfer learning with `torchvision.models.resnet18`, replaces the final fully connected layer for the target number of classes, and trains/evaluates with cross-entropy loss.

The default entrypoint is:
- `src/cnn.py` (runs training when executed directly)

---

## Features

- **Transfer Learning** with pretrained ResNet18 weights
- **Image Augmentation** for training robustness:
  - random resized crop
  - horizontal flip
  - rotation
  - color jitter
- **Standardized Normalization** using ImageNet mean/std
- **Validation-based Checkpoint Selection** (best model weights by validation accuracy)
- **Automated Model Saving** after training
- **Unit Tests** covering:
  - dataset dataframe creation
  - file extension filtering
  - dataset behavior and error handling
  - training/evaluation metric contracts
  - model output shape

---

## Repository Structure

```text
Brain-Tumor-Classification/
├── .github /workflows/
│   └── python-ci.yml          # GitHub Actions workflow (installs deps + runs tests)
├── Report/
│   └── report.pdf             # Project report
├── src/
│   ├── __init__.py
│   └── cnn.py                 # Main data, model, training, evaluation pipeline
├── test/
│   └── test_cnn.py            # Unit tests
├── pytest.ini                 # Pytest configuration
├── requirements.txt           # Python dependencies
├── script.sh                  # Helper script for env checks and running cnn.py
└── README.md
```

---

## How the Pipeline Works

`src/cnn.py` performs the following steps:

1. **Build DataFrames** from class-based directories:
   - `train_df(path)` and `test_df(path)` scan folders and collect image paths + class names.
2. **Map Class Labels**:
   - class names are mapped to integer labels.
3. **Create Datasets**:
   - `TumorDataset` loads image files and returns `(image_tensor_or_image, label_tensor)`.
4. **Split Training Set**:
   - 80% training / 20% validation split.
5. **Create DataLoaders**:
   - train, validation, and test loaders.
6. **Build Model**:
   - ResNet18 with a replaced final FC layer sized to `num_classes`.
7. **Train and Validate**:
   - train per epoch, evaluate on validation set, step LR scheduler.
8. **Keep Best Weights**:
   - saves the best validation-accuracy model state in memory.
9. **Final Test Evaluation**:
   - evaluates best model on test data.
10. **Save Weights**:
   - writes `resnet18_best_model.pth`.

---

## Dataset Layout

The training script expects this default directory structure:

```text
archive/
├── Training/
│   ├── class_1/
│   │   ├── image1.jpg
│   │   └── ...
│   ├── class_2/
│   └── ...
└── Testing/
    ├── class_1/
    ├── class_2/
    └── ...
```

Each subfolder name is treated as a class label.

Supported image extensions:
- `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`

---

## Installation

### 1) Clone the repository

```bash
git clone https://github.com/kgupta9191/Brain-Tumor-Classification.git
cd Brain-Tumor-Classification
```

### 2) Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
```

On Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

### 3) Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Quick Start

After preparing your dataset in `archive/Training` and `archive/Testing`:

```bash
python src/cnn.py
```

This will train the model and save:
- `resnet18_best_model.pth`

---

## Run Training

You can run training programmatically:

```python
from src.cnn import run_training

test_loss, test_acc = run_training(
    train_path="archive/Training",
    test_path="archive/Testing"
)
print(test_loss, test_acc)
```

If no device is provided, the script automatically selects:
- CUDA (if available)
- otherwise CPU

---

## Testing

Run tests with:

```bash
python -m pytest -q
```

`pytest.ini` is configured to discover tests in the `test/` directory.

---

## CI/CD

GitHub Actions workflow:
- File: `.github /workflows/python-ci.yml`
- Triggered on:
  - push to `main`
  - pull requests targeting `main`
  - manual dispatch
- Steps:
  1. checkout code
  2. setup Python 3.10
  3. install dependencies
  4. run pytest

---

## Configuration

Key hyperparameters in `src/cnn.py`:

- `batch_size = 256` (global constant; loaders currently use 32)
- `learning_rate = 0.001`
- `num_epochs = 20`
- `image_size = 256`

Current training optimizer/scheduler:
- Optimizer: `Adam`
- Loss: `CrossEntropyLoss`
- LR Scheduler: `StepLR(step_size=5, gamma=0.1)`

---

## Output Artifacts

After training:
- `resnet18_best_model.pth` — best model weights based on validation accuracy

---

## Troubleshooting

- **`No module named pytest`**  
  Install dependencies first: `python -m pip install -r requirements.txt`

- **Dataset path errors / no images found**  
  Verify directory names and class subfolders under `archive/Training` and `archive/Testing`.

- **CUDA not used**  
  Ensure PyTorch with CUDA support is installed and `torch.cuda.is_available()` returns `True`.

- **Out-of-memory issues**  
  Reduce dataloader batch size (currently set to 32 in loaders).

---

## Future Improvements

- Add experiment tracking (e.g., loss/accuracy logs per epoch)
- Add confusion matrix and classification report
- Add model inference script for single-image prediction
- Add reproducibility controls (seed handling and deterministic options)
- Add optional support for mixed precision training

---

## Contributing

Contributions are welcome.

Suggested process:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests locally
5. Open a pull request

---

## License

This project is licensed under the terms in [LICENSE](LICENSE).
