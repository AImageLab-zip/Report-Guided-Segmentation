# Medical Image Segmentation Framework

A flexible PyTorch-based framework for training 2D and 3D medical image segmentation models, with support for patch-based training, configurable architectures, and comprehensive metrics tracking.

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start and Usage Examples](#quick-start-and-usage-examples)
  - [Command Line Arguments](#command-line-arguments)
  - [Implement Your Own Training](#implement-your-own-training)
- [Detailed Components](#detailed-components)
  - [Base Classes](base/README.md)
  - [Configuration](config/README.md)
  - [Datasets](datasets/README.md)
  - [Losses](losses/README.md)
  - [Metrics](metrics/README.md)
  - [Models](models/README.md)
  - [Optimizers](optimizers/README.md)
  - [Trainers](trainer/README.md)
  - [Transforms](transforms/README.md)
  - [Utils](utils/README.md)
- [Notes](#notes)

## Project Structure

```
Brain-Segmentation/
├── base/                    # Abstract base classes
│   ├── base_dataset2d_sliced.py
│   ├── base_dataset3d.py
│   ├── base_model.py
│   └── base_trainer.py
├── config/                  # Configuration files for training and transforms
│   ├── config_atlas.json
│   ├── atlas_transforms.json
│   └── ...
├── datasets/                # Dataset loading and preprocessing (inherited from base_datasets)
│   ├── DatasetFactory.py
│   ├── ATLAS.py
│   └── BraTS2D.py
├── losses/                  # Loss function implementations
│   └── LossFactory.py
├── metrics/                 # Metrics computation and tracking
│   ├── MetricsFactory.py
│   └── MetricsManager.py
├── models/                  # Model architectures (inherited from base_model)
│   ├── ModelFactory.py
│   ├── UNet2D.py
│   └── UNet3D.py
├── optimizers/              # Optimizer configurations
│   └── OptimizerFactory.py
├── trainer/                 # Training logic (inherited from base_trainer)
│   ├── trainer_2Dsliced.py
│   └── trainer_3D.py
├── transforms/              # Data augmentation and preprocessing
│   └── TransformsFactory.py
├── utils/                   # Utility functions
│   ├── util.py
│   └── pad_unpad.py
├── config.py               # Config file handler
├── main.py                 # Training entry point
└── requirements.txt        # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kev98/Medical-Image-Segmentation.git
cd Medical-Image-Segmentation
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start and Usage Examples

The following are some base examples. You can add other CLI parameters useful for your main.py (which must be the entrypoint for training).

### Command Line Arguments

Command line arguments implemented in the provided main.py file:

- `--config`: Path to configuration JSON file (required)
- `--epochs`: Number of training epochs (required)
- `--save_path`: Directory to save model checkpoints (required)
- `--validation`: Enable validation during training (flag)
- `--resume`: Resume training from last checkpoint (flag)
- `--debug`: Enable debug mode with verbose output (flag)

Example of launch of main.py, training a 3D segmentation model, resuming checkpoints,

```bash
python main.py \
  --config config/config_atlas.json \
  --epochs 100 \
  --save_path /folder_containing_model_last.pth \
  --validation \
  --resume
```

### Implement Your Own Training

To set up a complete training pipeline, follow these steps:

- [**Create a Dataset Class**](#creating-a-new-dataset): Inherit from [BaseDataset3D](#base-dataset3dpy) or [BaseDataset2DSliced](#base-dataset2d-slicedpy) and implement the required abstract methods. 

- [**Implement a Model**](#adding-custom-models): Create your custom model by inheriting from [BaseModel](#base-modelpy) and implementing the `forward()` method.

- [**Implement a Trainer**](#implementing-a-custom-trainer): Create a custom trainer by inheriting from [BaseTrainer](#base-trainerpy) and implementing `_train_epoch()` and `eval_epoch()` methods.

- **Create an Entrypoint**: Write a `main.py` file that loads your configuration and instantiates your trainer. Use the provided [main.py](main.py) as a template or reference.

---

## Detailed Components

For detailed documentation on each component, refer to the README files in their respective directories:

- **[Base Classes](base/README.md)** - Abstract base classes for datasets, models, and trainers
- **[Configuration](config/README.md)** - JSON configuration files for training and transforms
- **[Datasets](datasets/README.md)** - Dataset loading and preprocessing
- **[Losses](losses/README.md)** - Loss function implementations
- **[Metrics](metrics/README.md)** - Metrics computation and tracking
- **[Models](models/README.md)** - Model architectures
- **[Optimizers](optimizers/README.md)** - Optimizer configurations
- **[Trainers](trainer/README.md)** - Training logic
- **[Transforms](transforms/README.md)** - Data augmentation and preprocessing
- **[Utils](utils/README.md)** - Utility functions

---

## Notes

- For patch-based training with 3D volumes, the framework uses TorchIO's Queue and GridSampler.
- Metrics are automatically computed per-class and averaged.
- Checkpoints are saved as `model_last.pth` and `model_best.pth` in the folder specified by the parameter --save_path.
- The framework is compatible with PyTorch 2.3+ and uses TorchIO's SubjectsLoader for proper data handling.