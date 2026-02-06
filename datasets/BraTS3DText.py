import os
import json
from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchio as tio

from base import BaseDataset


class BraTS3DText(BaseDataset):
    """
    BraTS3D dataset with optional text embeddings per subject.

    Uses folder "rep_RG" (instead of "vol"/"seg") and loads embeddings
    from the same folder with fixed .npy extension.
    """

    def __init__(self, config, root_folder, validation=True, train_transforms=None, test_transforms=None):
        self.report_folder = config.dataset.get("report_folder", None)
        super().__init__(config, root_folder, validation, train_transforms, test_transforms)

    def _load_text_embedding(self, report_path: str) -> Optional[torch.Tensor]:
        
        if not os.path.exists(report_path):
            return None

        data = np.load(report_path)
        if isinstance(data, np.lib.npyio.NpzFile):
            if "embedding" in data:
                array = data["embedding"]
            else:
                array = data[data.files[0]]
            return torch.from_numpy(array)

        return torch.from_numpy(data)
    
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

    def _get_subjects_list(self, split: str) -> List[tio.Subject]:
        images = getattr(self, f"{split}_images", [])
        labels = getattr(self, f"{split}_labels", [])
        subjects = []

        for image_path, label_path in zip(images, labels):
            subject_dir = {
                "image_path": image_path,
                "label_path": label_path,
                "image": tio.ScalarImage(image_path),
                "label": tio.LabelMap(label_path),
            }

            embedding = self._load_text_embedding(image_path.replace("/vol/", f"/{self.report_folder}/").replace("_vol.nii.gz", ".npz"))
            subject_dir["report"] = embedding

            subjects.append(tio.Subject(**subject_dir))

        print(f"Loaded {len(subjects)} subject for split {split}")
        return subjects

    def _filter_dataset_by_report(self, dataset: tio.SubjectsDataset, with_report: bool) -> tio.SubjectsDataset:
        subjects = dataset._subjects
        if with_report:
            filtered = [s for s in subjects if s.get("report") is not None]
        else:
            filtered = [s for s in subjects if s.get("report") is None]

        transform = getattr(dataset, "_transform", None)
        return tio.SubjectsDataset(filtered, transform=transform)
    
    def get_loader(self, split, report: bool = False):

        assert split in ['train', 'val', 'test'], 'Split must be train or val or test'

        if split == 'train':
            dataset = self._filter_dataset_by_report(self.train_set, with_report=report)
            return self._get_patch_loader(dataset, batch_size=self.config.dataset['batch_size'])
        elif split == 'val':
            if self.validation:
                dataset = self._filter_dataset_by_report(self.val_set, with_report=report)
                return self._get_entire_loader(dataset, batch_size=1)
            else:
                return
        elif split == 'test':
            dataset = self._filter_dataset_by_report(self.test_set, with_report=report)
            return self._get_entire_loader(dataset, batch_size=1)
