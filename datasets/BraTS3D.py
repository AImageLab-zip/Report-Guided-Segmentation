from base import BaseDataset
import os
from typing import Tuple, List
import json
from glob import glob
from pathlib import Path
import torch
import numpy as np
import torchio as tio

class AddTumorSamplingMap(tio.Transform):
    """
    Adds a sampling probability map:
      - label > 0  -> weight 5
      - label == 0 -> weight 1
    Stored as subject["sampling_map"] with type tio.SAMPLING_MAP.
    """
    def __init__(self):
        super().__init__()

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        label_img = subject["label"]      # tio.LabelMap
        lbl = label_img.data             # (1, D, H, W)

        tumor = (lbl > 0).to(torch.float32)
        prob = tumor * 5.0 + (1.0 - tumor) * 1.0

        subject["sampling_map"] = tio.Image(
            tensor=prob,
            affine=label_img.affine,
            type=tio.SAMPLING_MAP,
        )
        return subject

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

    
    def _get_patch_loader(self, dataset: tio.SubjectsDataset, batch_size: int = 1):
        use_weighted = bool(self.config.dataset.get("weighted_sampler", False))
        patch_size = self.config.dataset["patch_size"]

        if use_weighted:
            add_map = AddTumorSamplingMap()

            # TorchIO compatibility: use _transform (not transform)
            current_t = getattr(dataset, "_transform", None)

            if current_t is None:
                dataset._transform = add_map
            else:
                dataset._transform = tio.Compose([add_map, current_t])

            sampler = tio.WeightedSampler(
                patch_size=patch_size,
                probability_map="sampling_map",
            )
        else:
            sampler = tio.UniformSampler(patch_size=patch_size)

        queue = tio.Queue(
            subjects_dataset=dataset,
            max_length=self.config.dataset["queue_length"],
            samples_per_volume=self.config.dataset["samples_per_volume"],
            sampler=sampler,
            num_workers=self.NUM_WORKERS,
            shuffle_subjects=True,
            shuffle_patches=True,
            start_background=False,
        )

        loader = tio.SubjectsLoader(
            queue,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
        )
        return loader
    
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
            

