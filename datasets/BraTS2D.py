from typing import Tuple, List
import os
from glob import glob
import numpy as np
import nibabel as nib
from base.base_dataset2d_sliced import BaseDataset2DSliced
import torch
from tqdm import tqdm
import json
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
class BraTS2D(BaseDataset2DSliced):
    """
    BraTS dataset class for 2D slice extraction.
    
    BraTS dataset structure expected:
    root_folder/
    ├── imagesTr/
    │   ├── BraTS2021_00000_flair.nii.gz
    │   ├── BraTS2021_00000_t1.nii.gz
    │   ├── BraTS2021_00000_t1ce.nii.gz
    │   ├── BraTS2021_00000_t2.nii.gz
    │   └── ...
    ├── labelsTr/
    │   ├── BraTS2021_00000_seg.nii.gz
    │   └── ...
    ├── imagesTs/
    └── labelsTs/
    """
    
    def __init__(self, config, root_folder, validation=True, train_transforms=None, test_transforms=None, **kwargs):
        super().__init__(config, root_folder, validation, train_transforms, test_transforms, **kwargs)

    def _get_ordered_images_path(self) -> Tuple[List, List, List, List, List, List]:
        """
        Get the ordered images and their paths for BraTS dataset.
        """
        cases = os.listdir(self.root_folder)
        split_file = self.config.dataset.get("split_file", None)
        
        if split_file and os.path.exists(split_file):
            f = json.load(open(split_file, 'r'))
            train_cases = sorted(f.get("train", []))
            test_cases = sorted(f.get("test", []))
            val_cases = sorted(f.get("val", []))

            train_images = [os.path.join(self.root_folder, img) for img in train_cases if img in cases]
            test_images = [os.path.join(self.root_folder, img) for img in test_cases if img in cases]
            train_labels = [glob(os.path.join(self.root_folder, img, '*seg.nii.gz'))[0] for img in train_cases if img in cases]
            test_labels = [glob(os.path.join(self.root_folder, img, '*seg.nii.gz'))[0] for img in test_cases if img in cases]

            # Get val data
            if self.validation:
                val_images = [os.path.join(self.root_folder, img) for img in val_cases if img in cases]
                val_labels = [glob(os.path.join(self.root_folder, img, '*seg.nii.gz'))[0] for img in val_cases if img in cases]
            else:
                # Put also vaidation cases in the training set
                train_images += [os.path.join(self.root_folder, img) for img in val_cases if img in cases]
                train_labels += [glob(os.path.join(self.root_folder, img, '*seg.nii.gz'))[0] for img in val_cases if img in cases]
                train_images = sorted(train_images)
                train_labels = sorted(train_labels)
                val_images = []
                val_labels = []

        return train_images, train_labels, val_images, val_labels, test_images, test_labels

    def _extract_slice_indices(self, split: str) -> List[Tuple[int, int]]:
        slice_indices = []
        images = getattr(self, f'{split}_images', [])
        labels = getattr(self, f'{split}_labels', [])
        
        for vol_idx, (img_path, label_path) in tqdm(enumerate(zip(images, labels)), total=len(images), desc=f"Slicing {split} volumes"):
            try:
                label_img = nib.load(label_path)
                label_data = label_img.get_fdata()
                
                # Get number of slices along the specified axis
                num_slices = label_data.shape[self.slice_axis]
                
                for slice_idx in range(num_slices):
                    if self.slice_axis == 0: # sagittal
                        label_slice = label_data[slice_idx, :, :]
                    elif self.slice_axis == 1: # coronal
                        label_slice = label_data[:, slice_idx, :]
                    else: # axial
                        label_slice = label_data[:, :, slice_idx]
                    
                    # Check if slice has enough foreground
                    foreground_ratio = np.sum(label_slice > 0) / label_slice.size
                    
                    if foreground_ratio >= self.min_foreground_ratio:
                        slice_indices.append((vol_idx, slice_idx))
                        
            except Exception as e:
                print(f"Error processing volume {vol_idx} ({img_path}): {e}")
                continue
        
        return slice_indices
    
    def get_loader(self, split):
        assert split in ['train', 'val', 'test'], 'Split must be train or val or test'

        if split == 'train':
            return DataLoader(
                self.train_set,
                batch_size=self.config.dataset['batch_size'],
                shuffle=True,
                num_workers=self.NUM_WORKERS,
                pin_memory=False
            )
        elif split == 'val':
            if self.validation:
                return DataLoader(
                    self.val_set,
                    batch_size=1,
                    num_workers=self.NUM_WORKERS,
                    pin_memory=False
                )
            else:
                return
        elif split == 'test':
                return DataLoader(
                    self.test_set,
                    batch_size=1,
                    num_workers=self.NUM_WORKERS,
                    pin_memory=False
                )
    

    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for handling class imbalance.
        BraTS typically has classes: 0 (background), 1 (necrotic core), 
        2 (peritumoral edema), 4 (enhancing tumor)
        """

        class_counts = {}
        total_pixels = 0
        
        print("Calculating class weights...")
        for i, (volume_idx, slice_idx) in enumerate(self.slice_indices):
            if i % 100 == 0:
                print(f"Processed {i}/{len(self.slice_indices)} slices")
            
            label_path = self.label_paths[volume_idx]
            label_nii = nib.load(label_path)
            label_data = label_nii.get_fdata()
            
            # Extract slice
            if self.slice_axis == 0:
                label_slice = label_data[slice_idx, :, :]
            elif self.slice_axis == 1:
                label_slice = label_data[:, slice_idx, :]
            else:
                label_slice = label_data[:, :, slice_idx]
            
            # Count classes
            unique, counts = np.unique(label_slice, return_counts=True)
            for cls, count in zip(unique, counts):
                cls = int(cls)
                if cls not in class_counts:
                    class_counts[cls] = 0
                class_counts[cls] += count
                total_pixels += count
        
        # Calculate weights (inverse frequency)
        num_classes = max(class_counts.keys()) + 1
        weights = torch.ones(num_classes)
        
        for cls, count in class_counts.items():
            weights[cls] = total_pixels / (len(class_counts) * count)
        
        print(f"Class distribution: {class_counts}")
        print(f"Class weights: {weights}")
        
        return weights