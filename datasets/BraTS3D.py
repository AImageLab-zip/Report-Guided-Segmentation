from base import BaseDataset
import os
from typing import Tuple, List
import json
from glob import glob
from pathlib import Path
import torch
import numpy as np
import torchio as tio

class BraTS3D(BaseDataset):
    """
    Class for BraTS3D dataset creation and loading
    """
    def __init__(self, config, root_folder, validation=True, train_transforms=None, test_transforms=None):
        super().__init__(config, root_folder, validation, train_transforms, test_transforms)

    def _get_ordered_images_path(self) -> Tuple[List, List, List, List, List, List]:

        images_path = os.path.join(self.root_folder, 'vol')
        labels_path = os.path.join(self.root_folder, 'seg')
        images = glob(os.path.join(images_path, '*.nii.gz'))
        labels = glob(os.path.join(labels_path, '*.nii.gz'))

        # A split file must be specified, otherwise raise an error
        if self.config.dataset.get("split_file"):
            split_file = self.config.dataset.get("split_file")
            f = json.load(open(split_file, 'r'))
            
            train_images = sorted([i for i in images if Path(i).stem.removesuffix("_vol.nii") in f["train"]])
            train_labels = sorted([l for l in labels if Path(l).stem.removesuffix("_seg.nii") in f["train"]])
            val_images = sorted([i for i in images if Path(i).stem.removesuffix("_vol.nii") in f["val"]])
            val_labels = sorted([l for l in labels if Path(l).stem.removesuffix("_seg.nii") in f["val"]])
            test_images = sorted([i for i in images if Path(i).stem.removesuffix("_vol.nii") in f["test"]])
            test_labels = sorted([l for l in labels if Path(l).stem.removesuffix("_seg.nii") in f["test"]])

        else:
            raise ValueError("A split file must be specified for BraTS3D dataset when validation is True.")

        if not self.validation:
            train_images = train_images + val_images
            train_labels = train_labels + val_labels

        return train_images, train_labels, val_images, val_labels, test_images, test_labels

    
    def get_loader(self, split):

        assert split in ['train', 'val', 'test'], 'Split must be train or val or test'

        if split == 'train':
            return self._get_patch_loader(self.train_set, batch_size=self.config.dataset['batch_size'])
        elif split == 'val':
            if self.validation:
                return self._get_entire_loader(self.val_set, batch_size=1)
            else:
                return
        elif split == 'test':
            return self._get_entire_loader(self.test_set, batch_size=1)
            

