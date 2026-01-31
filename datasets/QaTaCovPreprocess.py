from __future__ import annotations

import os
from typing import List, Tuple, Optional, Dict

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from base.base_dataset import BaseDataset
from transforms import QaTaCov as QaTaCovTransforms


class QaTaCovPreprocess(BaseDataset):
    """
    QaTa-COV19-v2 dataset for 2D chest X-ray segmentation (raw/preprocess stage).

    Expected folder layout under root_folder:

    root_folder/
        Train Set/
            Images/
            Ground-truths/
            Text/
            Train_ID.xlsx
            Val_ID.xlsx
        Test Set/
            Images/
            Ground-truths/
            Text/

    Train/Val split is defined by Train_ID.xlsx and Val_ID.xlsx, which contain
    a column named "Image" listing mask filenames (e.g., mask_covid_1.png).
    """

    def __init__(self, config, root_folder, validation=True, train_transforms=None, test_transforms=None):
        self.config = config
        self.validation = validation
        self.root_folder = root_folder
        self.train_images, self.train_labels, self.val_images, self.val_labels, self.test_images, self.test_labels = self._get_ordered_images_path()

        image_size = self.config.dataset.get('image_size', [224, 224])
        train_transforms = QaTaCovTransforms(mode='train').transform(image_size=image_size)
        test_transforms = QaTaCovTransforms(mode='test').transform(image_size=image_size)

        train_text_dir = os.path.join(self.root_folder, 'Train Set', 'Text')
        test_text_dir = os.path.join(self.root_folder, 'Test Set', 'Text')

        train_text_map = self._load_text_map(train_text_dir)
        test_text_map = self._load_text_map(test_text_dir)

        self.train_set = QaTaCov2DSet(
            self.train_images,
            self.train_labels,
            train_text_map,
            train_transforms
        )
        if self.validation:
            self.val_set = QaTaCov2DSet(
                self.val_images,
                self.val_labels,
                train_text_map,
                test_transforms
            )
        self.test_set = QaTaCov2DSet(
            self.test_images,
            self.test_labels,
            test_text_map,
            test_transforms
        )

    def _get_ordered_images_path(self) -> Tuple[List, List, List, List, List, List]:
        train_root = os.path.join(self.root_folder, 'Train Set')
        test_root = os.path.join(self.root_folder, 'Test Set')

        train_images_dir = os.path.join(train_root, 'Images')
        train_masks_dir = os.path.join(train_root, 'Ground-truths')
        test_images_dir = os.path.join(test_root, 'Images')
        test_masks_dir = os.path.join(test_root, 'Ground-truths')

        train_ids_path = os.path.join(train_root, 'Train_ID.xlsx')
        val_ids_path = os.path.join(train_root, 'Val_ID.xlsx')

        train_ids = self._load_id_list(train_ids_path)
        val_ids = self._load_id_list(val_ids_path)

        train_images = [os.path.join(train_images_dir, self._mask_to_image_name(mask)) for mask in train_ids]
        train_labels = [os.path.join(train_masks_dir, mask) for mask in train_ids]

        if self.validation:
            val_images = [os.path.join(train_images_dir, self._mask_to_image_name(mask)) for mask in val_ids]
            val_labels = [os.path.join(train_masks_dir, mask) for mask in val_ids]
        else:
            val_images = []
            val_labels = []

        test_labels = sorted([f for f in os.listdir(test_masks_dir) if f.lower().endswith('.png')])
        test_images = [os.path.join(test_images_dir, self._mask_to_image_name(mask)) for mask in test_labels]
        test_labels = [os.path.join(test_masks_dir, mask) for mask in test_labels]

        return train_images, train_labels, val_images, val_labels, test_images, test_labels

    def _load_id_list(self, excel_path: str) -> List[str]:
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Split file not found: {excel_path}")
        df = pd.read_excel(excel_path)
        if 'Image' not in df.columns:
            raise ValueError(f"Expected column 'Image' in {excel_path}, found {df.columns.tolist()}")
        return df['Image'].dropna().astype(str).tolist()

    def _load_text_map(self, text_dir: str) -> Dict[str, str]:
        if not os.path.isdir(text_dir):
            return {}

        excel_files = [
            f for f in os.listdir(text_dir)
            if f.lower().endswith('.xlsx')
        ]
        if not excel_files:
            return {}

        text_map: Dict[str, str] = {}
        for excel_file in excel_files:
            df = pd.read_excel(os.path.join(text_dir, excel_file))
            if 'Image' not in df.columns or 'Description' not in df.columns:
                continue
            for _, row in df[['Image', 'Description']].dropna().iterrows():
                text_map[str(row['Image'])] = str(row['Description'])

        return text_map

    @staticmethod
    def _mask_to_image_name(mask_name: str) -> str:
        if mask_name.startswith('mask_'):
            return mask_name.replace('mask_', '', 1)
        return mask_name

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


class QaTaCov2DSet(Dataset):
    """
    Dataset returning image/mask pairs (and optional text) for QaTaCov.
    """

    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        text_map: Optional[Dict[str, str]] = None,
        transform=None,
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.text_map = text_map or {}
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        data = {'image': image_path, 'gt': mask_path}
        if self.transform is not None:
            data = self.transform(data)

        image = data['image']
        gt = data['gt']
        #gt = torch.where(gt == 255, 1, 0)

        sample = {
            'image': image,
            'label': gt,
            'image_path': image_path,
            'label_path': mask_path,
        }

        if self.text_map:
            mask_name = os.path.basename(mask_path)
            sample['text'] = self.text_map.get(mask_name, '')

        return sample

    def _image_to_text_path(self, image_path: str) -> str:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        return f"{base_name}.txt"