import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Callable, TypedDict
import torchio as tio

# Si potrebbe fare una classe astratta che fa vedere quali metodi devono essere implementati, che sono usati sempre dal trainer
class BaseDataset2DSliced:
    """
    Base class for 2D datasets extracted from 3D medical images, in a nifti format
    Each 2D slice becomes a separate sample in the dataset.
    """

    if 'SLURM_CPUS_PER_TASK' in os.environ:
        NUM_WORKERS = int(os.environ['SLURM_CPUS_PER_TASK'])
        print(f'Detected {NUM_WORKERS} cpus by 2d sliced dataset')
    else:
        NUM_WORKERS = 4
        print(f'Number of workers set to {NUM_WORKERS} cpus')

    def __init__(self, config, root_folder, validation=True, train_transforms=None, test_transforms=None, **kwargs):
        """
        Args:
            config: Configuration object containing dataset parameters
            root_folder: Root directory containing the dataset
            split: Dataset split ('train', 'val', 'test')
            validation: Whether to create validation split
            transform: Transformations to apply to 2D slices
            slice_axis: Axis along which to extract slices (0=sagittal, 1=coronal, 2=axial)
            min_foreground_ratio: Minimum ratio of foreground pixels to include a slice
        """
        self.config = config
        self.root_folder = root_folder
        self.validation = validation
        self.modalities = config.dataset.get('modalities', [])
        self.slice_axis = config.dataset.get('slice_axis', 2)
        self.min_foreground_ratio = config.dataset.get('min_foreground_ratio', 0.01)

        self.train_images, self.train_labels, self.val_images, self.val_labels, self.test_images, self.test_labels = self._get_ordered_images_path()
        assert (len(self.train_images) == len(self.train_labels) and
                len(self.val_images) == len(self.val_labels) and
                len(self.test_images) == len(self.test_labels)), "Mismatch in data lengths"
        
        # Extract all valid 2D slices
        train_slice_indices = self._extract_slice_indices("train")
        if self.validation:
            val_slices_indices = self._extract_slice_indices("val")
        test_slices_indices = self._extract_slice_indices("test")

        #print(f"Loaded {len(self.slice_indices)} 2D slices for {split} split")

        self.train_set = BaseSet2D(self.train_images, self.train_labels, train_slice_indices, self.modalities, train_transforms, self.slice_axis, self.min_foreground_ratio)
        if self.validation:
           self.val_set = BaseSet2D(self.val_images, self.val_labels, val_slices_indices, self.modalities, test_transforms, self.slice_axis, self.min_foreground_ratio)
        self.test_set = BaseSet2D(self.train_images, self.train_labels, test_slices_indices, self.modalities, train_transforms, self.slice_axis, self.min_foreground_ratio)

        

    def _get_ordered_images_path(self) -> Tuple[List, List, List, List, List, List]:
        """
        Get the ordered images and their paths for train, val, and test splits.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def _extract_slice_indices(self, split) -> List[Tuple[int, int]]:
        """
        Extract valid slice indices from all volumes.
        Returns: list of (volume_idx, slice_idx) tuples.
        """
        raise NotImplementedError

    def get_loader(self, split) -> DataLoader:
        """
        Get the correct Dataloader based on the phase (train, validation, test).

        :param split:
        :return: One of the Dataloader implemented above
        """
        
    
class BaseSet2D(Dataset):
    """
    Base class for 2D datasets extracted from 3D medical images, in a nifti format
    Each 2D slice becomes a separate sample in the dataset.
    """
    def __init__(self, images_paths, labels_paths, slice_indices, modalities, transform=None, slice_axis=2, min_foreground_ratio=0.01):
        super().__init__()

        self.images_paths = images_paths
        self.labels_paths = labels_paths
        self.slice_indices = slice_indices # List of tuples (volume_idx, slice_idx), actual number of set samples
        self.transform = transform
        self.slice_axis = slice_axis
        self.min_foreground_ratio = min_foreground_ratio
        self.modalities = modalities


    def __len__(self) -> int:
        return len(self.slice_indices)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single 2D slice.
        
        Returns:
            dict: Dictionary containing 'image', 'label', 'volume_idx', 'slice_idx'
        """
        volume_idx, slice_idx = self.slice_indices[idx]
        modality_slices = [] 

        for m in self.modalities:
            img_path = self.images_paths[volume_idx]
            img_path = os.path.join(img_path, img_path.split("/")[-1] + f"-{m}.nii.gz")
            img_nii = nib.load(img_path)
            img_data = img_nii.get_fdata()

            # Extract 2D slice for current modality
            if self.slice_axis == 0:  # sagittal
                img_slice = img_data[slice_idx, :, :]
            elif self.slice_axis == 1:  # coronal
                img_slice = img_data[:, slice_idx, :]
            else:  # axial (default)
                img_slice = img_data[:, :, slice_idx]

            modality_slices.append(img_slice)

        # Stack all modalities into a multi-channel 2D image
        image_slice = np.stack(modality_slices, axis=0)  # Shape: (C, H, W)

        label_path = self.labels_paths[volume_idx] 
        label_nii = nib.load(label_path)
        label_data = label_nii.get_fdata()

        if self.slice_axis == 0:  # sagittal
                label_slice = label_data[slice_idx, :, :]
        elif self.slice_axis == 1:  # coronal
            label_slice = label_data[:, slice_idx, :]
        else:  # axial (default)
            label_slice = label_data[:, :, slice_idx]

        image_tensor = torch.from_numpy(image_slice.astype(np.float32))
        label_tensor = torch.from_numpy(label_slice.astype(np.int64))

        if self.transform:
            if image_tensor.ndim == 3:  # [C, H, W]
                image_tensor = image_tensor.unsqueeze(1)  # [C, 1, H, W]
            if label_tensor.ndim == 2:  # [H, W]
                label_tensor = label_tensor.unsqueeze(0).unsqueeze(1)  # [1, 1, H, W]

            subject = tio.Subject(
                image=tio.ScalarImage(tensor=image_tensor),
                label=tio.LabelMap(tensor=label_tensor)
            )

            transformed = self.transform(subject)

            image_tensor = transformed['image'].data.squeeze(1)  # [C, H, W]
            label_tensor = subject['label'].data.squeeze(0).squeeze(0)  # [H, W]

        sample = {
            'image': image_tensor,
            'label': label_tensor,
            'volume_idx': volume_idx,
            'slice_idx': slice_idx,
            'image_path': img_path,
            'label_path': label_path
        }
        
        return sample