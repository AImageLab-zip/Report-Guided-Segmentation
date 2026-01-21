# Datasets

## 📁 [`DatasetFactory.py`](DatasetFactory.py)
Factory pattern for creating dataset instances from config. The factory searches for dataset classes in the `datasets/` folder by looking up the dataset name in the module's namespace.

## Creating a New Dataset

**For 2D/3D datasets (inherit from [BaseDataset](/base/base_dataset.py)):**

Required methods:
1. `_get_ordered_images_path()` → Returns `(train_images, train_labels, val_images, val_labels, test_images, test_labels)`
2. `get_loader(split)` → Returns appropriate loader for 'train'/'val'/'test'

Optional: Override `_get_patch_loader()` if you need a custom weighted sampler for patch-based training.

**For 2D sliced datasets (inherit from [BaseDataset2DSliced](/base/base_dataset2d_sliced.py)):**

Required methods:
1. `_get_ordered_images_path()` → Returns paths to 3D volumes
2. `_extract_slice_indices(split)` → Returns list of (volume_idx, slice_idx) tuples
3. `get_loader(split)` → Returns DataLoader for 2D slices

**Examples:**
- [ATLAS.py](/datasets/ATLAS.py): 3D segmentation dataset with optional train/val split files or random splitting
- [BraTS2D.py](/datasets/BraTS2D.py): 2D sliced version extracting axial slices from 3D BraTS volumes


### Important Note

**When you create a new dataset class, remember to add it to [`__init__.py`](__init__.py) in the `datasets/` folder.** This ensures the DatasetFactory can find and instantiate your custom dataset class. Simply import your new dataset class in the `__init__.py` file:

```python
from .YourNewDataset import YourNewDataset
```
