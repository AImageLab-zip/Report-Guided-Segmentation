from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from base.base_dataset import BaseDataset


class QaTaCov(BaseDataset):
    """
    QaTaCov dataset for 2D chest X-ray segmentation using preprocessed data.

    Expected folder layout under root_folder:

    root_folder/
        Images/
        Ground-truths/
        qatacov_split.xlsx

    qatacov_split.xlsx columns:
        - Image: filename of the image (e.g., covid_1.png)
        - Split: train/val/test
        - Report: text report associated with the sample

    Ground-truths are expected to use the naming convention: mask_<image_name>.
    """

    def __init__(self, config, root_folder, validation=True, train_transforms=None, test_transforms=None):
        self.config = config
        self.validation = validation
        self.root_folder = root_folder
        self.train_images, self.train_labels, self.val_images, self.val_labels, self.test_images, self.test_labels = self._get_ordered_images_path()

        metadata = self._load_metadata()

        self.train_set = QaTaCov2DSet(
            self.train_images,
            self.train_labels,
            metadata.get('train', {}),
            train_transforms
        )
        if self.validation:
            self.val_set = QaTaCov2DSet(
                self.val_images,
                self.val_labels,
                metadata.get('val', {}),
                test_transforms
            )
        self.test_set = QaTaCov2DSet(
            self.test_images,
            self.test_labels,
            metadata.get('test', {}),
            test_transforms
        )

    def _load_metadata(self) -> Dict[str, Dict[str, str]]:
        split_file = self.config.dataset.get('split_file', None)
        if not split_file:
            split_file = os.path.join(self.root_folder, 'qatacov_split.xlsx')
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

        df = pd.read_excel(split_file)
        required = {'Image', 'Split', 'Report'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in {split_file}: {sorted(missing)}")

        metadata: Dict[str, Dict[str, str]] = {'train': {}, 'val': {}, 'test': {}}
        for _, row in df[['Image', 'Split', 'Report']].dropna().iterrows():
            split = str(row['Split']).strip().lower()
            image_name = str(row['Image']).strip()
            if split in metadata:
                metadata[split][image_name] = str(row['Report'])
        return metadata

    def _get_ordered_images_path(self) -> Tuple[List, List, List, List, List, List]:
        images_dir = os.path.join(self.root_folder, 'Images')
        masks_dir = os.path.join(self.root_folder, 'Ground-truths')

        split_file = self.config.dataset.get('split_file', None)
        if not split_file:
            split_file = os.path.join(self.root_folder, 'qatacov_split.xlsx')
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

        df = pd.read_excel(split_file)
        required = {'Image', 'Split'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in {split_file}: {sorted(missing)}")

        def build_paths(split: str) -> Tuple[List[str], List[str]]:
            split_df = df[df['Split'].astype(str).str.lower() == split]
            images = []
            labels = []
            for image_name in split_df['Image'].dropna().astype(str).tolist():
                image_path = os.path.join(images_dir, image_name)
                mask_name = f"mask_{image_name}"
                mask_path = os.path.join(masks_dir, mask_name)
                if os.path.exists(image_path) and os.path.exists(mask_path):
                    images.append(image_path)
                    labels.append(mask_path)
            return images, labels

        train_images, train_labels = build_paths('train')
        if self.validation:
            val_images, val_labels = build_paths('val')
        else:
            val_images, val_labels = [], []
        test_images, test_labels = build_paths('test')

        return train_images, train_labels, val_images, val_labels, test_images, test_labels

    def get_loader(self, split: str):
        assert split in ['train', 'val', 'test'], 'Split must be train or val or test'

        if split == 'train':
            return DataLoader(
                self.train_set,
                batch_size=self.config.dataset['batch_size'],
                shuffle=True,
                num_workers=self.NUM_WORKERS,
                pin_memory=False
            )
        if split == 'val':
            if self.validation:
                return DataLoader(
                    self.val_set,
                    batch_size=1,
                    num_workers=self.NUM_WORKERS,
                    pin_memory=False
                )
            return
        return DataLoader(
            self.test_set,
            batch_size=1,
            num_workers=self.NUM_WORKERS,
            pin_memory=False
        )
    

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights from 2D binary masks.
        Assumes lesion pixels are >0 in the stored mask PNG.
        Returns weights tensor of shape [2]: [w_bg, w_lesion]
        """
        class_counts = {0: 0, 1: 0}
        total_pixels = 0

        print("Calculating class weights from masks...")
        for i, mask_path in enumerate(self.mask_paths):
            if i % 200 == 0:
                print(f"Processed {i}/{len(self.mask_paths)} masks")

            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
            lesion = (mask > 0).astype(np.uint8)  # 0/1

            num_lesion = int(lesion.sum())
            num_bg = int(lesion.size - num_lesion)

            class_counts[0] += num_bg
            class_counts[1] += num_lesion
            total_pixels += lesion.size

        # inverse frequency (normalized by number of classes)
        num_classes = 2
        weights = torch.ones(num_classes, dtype=torch.float32)

        for cls in [0, 1]:
            count = max(1, class_counts[cls])
            weights[cls] = total_pixels / (num_classes * count)

        print(f"Class distribution (pixels): {class_counts}")
        print(f"Class weights: {weights.tolist()}")

        return weights


class QaTaCov2DSet(Dataset):
    """
    Dataset returning image/mask pairs (and optional report text) for QaTaCov.
    """

    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        report_map: Optional[Dict[str, str]] = None,
        transform=None,
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.report_map = report_map or {}
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image_tensor = torch.from_numpy(
            np.array(Image.open(image_path).convert("L"), dtype=np.float32)
        ).unsqueeze(0)  # [1,H,W]

        label_tensor = torch.from_numpy(
            np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        ).unsqueeze(0)  # [1,H,W]

        
        sample = {"image": image_tensor, "label": label_tensor}

        if self.transform is not None:
            sample = self.transform(sample)

        image_tensor = sample["image"]
        gt = sample["label"]


        gt = (gt > 0).long()

        sample = {
            "image": image_tensor,   # float in [0,1]
            "label": gt,             # float in {0,1}
            "image_path": image_path,
            "label_path": mask_path,
        }

        if self.report_map:
            image_name = os.path.basename(image_path)
            sample["text"] = self.report_map.get(image_name, "")

        return sample
