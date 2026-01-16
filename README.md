# Brain Segmentation Framework

A flexible PyTorch-based framework for training 2D and 3D medical image segmentation models, with support for patch-based training, configurable architectures, and comprehensive metrics tracking.

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start-and-usage-examples)
- [Detailed Components](#detailed-components)
  - [Base Classes](#base-classes)
  - [Configuration](#configuration)
  - [Datasets](#datasets)
  - [Losses](#losses)
  - [Metrics](#metrics)
  - [Models](#models)
  - [Optimizers](#optimizers)
  - [Trainers](#trainers)
  - [Transforms](#transforms)
  - [Utils](#utils)
- [Creating a New Dataset](#creating-a-new-dataset)
- [Usage Examples](#usage-examples)

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

- [**Implement a Model**](#adding-custom-models): Create your custom model by inheriting from [BaseModel](#base-modelpy) and implementing the `forward()` method. Refer to [Adding Custom Models](#adding-custom-models) for guidance.

- [**Implement a Trainer**](#implementing-a-custom-trainer): Create a custom trainer by inheriting from [BaseTrainer](#base-trainerpy) and implementing `_train_epoch()` and `eval_epoch()` methods.

- **Create an Entrypoint**: Write a `main.py` file (or similar) that loads your configuration and instantiates your trainer. Use the provided [main.py](main.py) as a template or reference.





## Detailed Components

### Base Classes

The `base/` directory contains abstract base classes that define the core interfaces for datasets, models, and trainers. These classes are designed to be inherited and customized for your specific use case, providing a consistent architecture across the framework.

#### 💭 `base/base_dataset3d.py`
Base class for 3D volume datasets, which Loads train/validation/test image and label paths, creates TorchIO `SubjectsDataset` for each split, and initialize all the other component needed to process the dataset.

**Abstract Methods (Must Implement):**
- `_get_ordered_images_path()`: Return 6 lists of paths, corresponding train images/labels, val images/labels, test images/labels paths (val lists can be none if you don't use --validation)
- `get_loader(split)`: Return appropriate loader for 'train'/'val'/'test' split

**Implemented Methods (Provided):**
- `_get_subjects_list(split)`: Creates TorchIO Subject objects from image/label paths
- `_get_patch_loader(dataset)`: Returns a patch-based loader using TorchIO Queue for training
- `_get_volume_loader(dataset)`: Returns a loader for entire volumes (used in validation/testing)

#### 💭 `base/base_dataset2d_sliced.py`
Base class for 2D slice-based training from 3D volumes, which loads train/validation/test image and label paths, creates TorchIO `SubjectsDataset` of slices from the 3D dataset, and initialize all the other component needed to process the dataset.

**Abstract Methods (Must Implement):**
- `_get_ordered_images_path()`: Return paths for 3D volumes
- `_extract_slice_indices(split)`: Extract valid 2D slices from 3D volumes and return list of (volume_idx, slice_idx) tuples
- `get_loader(split)`: Return DataLoader for 2D slices

**Implemented Methods (Provided):**
- Constructor handles train/val/test split initialization using the slice indices
- `BaseSet2D` dataset wrapper for handling 2D slices from 3D volumes

#### 💭 `base/base_model.py`
Simple wrapper around `nn.Module` for consistency. Inherit from this when creating custom models.

**Abstract Methods (Must Implement):**
- `forward(*inputs)`: Define the forward pass logic for your model

**Implemented Methods (Provided):**
- `__str__()`: Displays model summary with trainable parameter count

#### 💭 `base/base_trainer.py`
Base trainer class handling:
- Model, optimizer, scheduler, loss, and metrics initialization
- Dataset loading with transforms
- Checkpoint saving/resuming
- Training loop orchestration

**Abstract Methods (Must Implement):**
- `_train_epoch(epoch)`: Training logic for one epoch
- `eval_epoch(epoch, phase)`: Validation/test logic

**Implemented Methods (Provided):**
- Constructor initialization of model, optimizer, scheduler, loss, and metrics from config
- `train()`: Main training loop that orchestrates epochs and checkpoint management
- Checkpoint save/resume functionality
- Metrics tracking and results aggregation

--- 
### Configuration

#### 📚 `config/`
Contains JSON configuration files organized in two types:

**1. General Config (e.g., `config_atlas.json`):**
Defines model, dataset, training parameters, optimizer, loss, and metrics.

**2. Transforms Config (e.g., `atlas_transforms.json`):**
Specifies preprocessing and augmentation pipelines using TorchIO transforms (to be passed in the general config under the key dataset.transforms as shown below).

**Structure:**
```json
{
  "name": "experiment_name",
  "model": {"type": "UNet3D", "params": {...}},
  "dataset": {
    "type": "ATLAS",
    "root_folder": "/path/to/data",
    "transforms": "config/atlas_transforms.json",
    ...
  },
  "optimizer": {"type": "Adam", "args": {...}},
  "loss": {"name": "DiceLoss", "loss_kwargs": {...}},
  "metrics": {"name": ["DiceMetric"]}
}
```

The transforms file is referenced in `dataset.transforms` and contains:
```json
{
  "preprocessing": [...],
  "augmentations": [...]
}
```
---
### Datasets

#### 📁 `datasets/DatasetFactory.py`
Factory pattern for creating dataset instances from config.

#### Creating a New Dataset

**For 3D datasets (inherit from `BaseDataset3D`):**

Required methods:
1. `_get_ordered_images_path()` → Returns `(train_images, train_labels, val_images, val_labels, test_images, test_labels)`
2. `get_loader(split)` → Returns appropriate loader for 'train'/'val'/'test'

Optional: Override `_get_patch_loader()` if you need a custom weighted sampler for patch-based training.

**For 2D sliced datasets (inherit from `BaseDataset2DSliced`):**

Required methods:
1. `_get_ordered_images_path()` → Returns paths to 3D volumes
2. `_extract_slice_indices(split)` → Returns list of (volume_idx, slice_idx) tuples
3. `get_loader(split)` → Returns DataLoader for 2D slices

**Examples:**
- `ATLAS.py`: 3D segmentation dataset with optional train/val split files or random splitting
- `BraTS2D.py`: 2D sliced version extracting axial slices from 3D BraTS volumes

---

### Losses

#### 🎯 `losses/LossFactory.py`
Creates loss functions from config. Supports MONAI losses out-of-the-box.

**Adding Custom Losses:**
1. Create a new file in `losses/` with your loss class
2. Specify the class name in config: `"loss": {"name": "YourLoss", "loss_kwargs": {...}}`

---

### Metrics

#### 📊 `metrics/MetricsFactory.py`
Creates metric instances (supports MONAI metrics).

#### 📊 `metrics/MetricsManager.py`
Handles metric computation, accumulation, and storage.

**Functionality:**
- Computes metrics per iteration and accumulates them
- Handles both single-value metrics (losses) and multi-class metrics
- Automatically computes per-class and mean metrics
- Stores results in pandas DataFrame with one row per epoch

**Saved Format:**
CSV files (`train_metrics.csv`, `val_metrics.csv`, `test_metrics.csv`) with columns:
- `epoch`: Epoch number
- `<metric_name>`: For scalar metrics
- `<metric_name>_<class_name>`: For per-class metrics
- `<metric_name>_mean`: Mean across all classes

### Models

#### 🧠 `models/ModelFactory.py`
Creates model instances from config.

#### Available Models

🧠 **`UNet2D.py`:**
- 2D U-Net architecture
- Configurable: `in_channels`, `num_classes`, `init_features`

🧠 **`UNet3D.py`:**
- 3D U-Net architecture  
- Configurable: `in_channels`, `num_classes`, `init_features`

**Adding Custom Models:**
1. Create a new file in `models/`
2. Inherit from `BaseModel` (or directly from `nn.Module`)
3. Implement `__init__()` to define layers
4. Implement `forward()` method
5. Specify in config: `"model": {"type": "YourModel", "params": {...}}`

---
### Optimizers

#### 📈 `optimizers/OptimizerFactory.py`
Creates PyTorch optimizers and learning rate schedulers from config.

---
### Trainers

#### 🏋🏼‍♂️ `trainer/trainer_3D.py`
Trainer for 3D volume segmentation.

**Key Methods:**
- `_train_epoch(epoch)`: Iterates through patch-based training loader, computes loss, backpropagates, updates metrics
- `eval_epoch(epoch, phase)`: Performs patch-based inference on full volumes using GridSampler/GridAggregator, computes validation/test metrics
- `_inference_sampler(sample)`: Creates GridSampler for dense patch-based inference
- `_results_dict(phase, epoch)`: Retrieves metrics for the epoch

#### 🏋🏼‍♂️`trainer/trainer_2Dsliced.py`
Trainer for 2D slice-based segmentation.

**Implementing a Custom Trainer:**
1. Inherit from `BaseTrainer`
2. Implement `_train_epoch(epoch)`: Define training loop, return metrics dict
3. Implement `eval_epoch(epoch, phase)`: Define validation/test loop, return metrics dict

---
### Transforms

#### 🌀 `transforms/TransformsFactory.py`
Creates TorchIO transform pipelines from JSON config.

**Supported:**
- TorchIO preprocessing transforms (resampling, cropping, etc.)
- TorchIO augmentation transforms (flip, affine, noise, etc.)

**Custom Transforms:**
Alternatively, implement preprocessing/augmentation directly in your dataset class.

---
### Utils

#### 🛠️ `utils/util.py`
Helper functions including one-hot encoding.

#### 🛠️ `utils/pad_unpad.py`
Functions for padding/unpadding operations.

## Notes

- For patch-based training with 3D volumes, the framework uses TorchIO's Queue and GridSampler.
- Metrics are automatically computed per-class and averaged.
- Checkpoints are saved as `model_last.pth` and `model_best.pth` in the folder specified by the parameter --save_path.
- The framework is compatible with PyTorch 2.3+ and uses TorchIO's SubjectsLoader for proper data handling.