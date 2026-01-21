# Base Classes

The `base/` directory contains abstract base classes that define the core interfaces for datasets, models, and trainers. These classes are designed to be inherited and customized for your specific use case, providing a consistent architecture across the framework.

## 💭 [base_dataset3d.py](base_dataset3d.py)
Base class for 3D volume datasets, which Loads train/validation/test image and label paths, creates TorchIO `SubjectsDataset` for each split, and initialize all the other component needed to process the dataset.

**Abstract Methods (Must Implement):**
- `_get_ordered_images_path()`: Return 6 lists of paths, corresponding train images/labels, val images/labels, test images/labels paths (val lists can be none if you don't use --validation)
- `get_loader(split)`: Return appropriate loader for 'train'/'val'/'test' split

**Implemented Methods (Provided):**
- `_get_subjects_list(split)`: Creates TorchIO Subject objects from image/label paths
- `_get_patch_loader(dataset)`: Returns a patch-based loader using TorchIO Queue for training
- `_get_entire_loader(dataset)`: Returns a loader for entire volume/image

## 💭 [base_dataset2d_sliced.py](base_dataset2d_sliced.py)
Base class for 2D slice-based training from 3D volumes, which loads train/validation/test image and label paths, creates TorchIO `SubjectsDataset` of slices from the 3D dataset, and initialize all the other component needed to process the dataset.

**Abstract Methods (Must Implement):**
- `_get_ordered_images_path()`: Return paths for 3D volumes
- `_extract_slice_indices(split)`: Extract valid 2D slices from 3D volumes and return list of (volume_idx, slice_idx) tuples
- `get_loader(split)`: Return DataLoader for 2D slices

**Implemented Methods (Provided):**
- Constructor handles train/val/test split initialization using the slice indices
- `BaseSet2D` dataset wrapper for handling 2D slices from 3D volumes

## 💭 [base_model.py](base_model.py)
Simple wrapper around `nn.Module` for consistency. Inherit from this when creating custom models.

**Abstract Methods (Must Implement):**
- `forward(*inputs)`: Define the forward pass logic for your model

**Implemented Methods (Provided):**
- `__str__()`: Displays model summary with trainable parameter count

## 💭 [base_trainer.py](base_trainer.py)
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
