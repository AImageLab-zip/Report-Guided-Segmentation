from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from base.base_dataset import BaseDataset
from datasets.QaTaCov import QaTaCov2DSet


class QaTaCovTextEmb(BaseDataset):
    """
    QaTaCov dataset for 2D chest X-ray segmentation with precomputed report embeddings.

    Expected folder layout under root_folder:

    root_folder/
        Images/
        Ground-truths/
        qatacov_split.xlsx

    Text embeddings layout (config.dataset.text_emb_dir):
        reports_emb.npy
        image_to_report_idx.json
    """

    def __init__(self, config, root_folder, validation=True, train_transforms=None, test_transforms=None):
        self.config = config
        self.validation = validation
        self.root_folder = root_folder
        self.train_images, self.train_labels, self.val_images, self.val_labels, self.test_images, self.test_labels = self._get_ordered_images_path()

        metadata = self._load_metadata()
        text_emb_dir = self._get_text_emb_dir()
        reports_emb, image_to_idx = self._load_text_embeddings(text_emb_dir)

        self.train_set = QaTaCov2DTextEmbSet(
            self.train_images,
            self.train_labels,
            metadata.get("train", {}),
            reports_emb,
            image_to_idx,
            train_transforms,
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

    def _get_text_emb_dir(self) -> str:
        text_emb_dir = self.config.dataset.get("text_emb_dir")
        if not text_emb_dir:
            raise ValueError("config.dataset.text_emb_dir is required for QaTaCovTextEmb")
        if not os.path.isdir(text_emb_dir):
            raise FileNotFoundError(f"Text embedding directory not found: {text_emb_dir}")
        return text_emb_dir

    def _load_text_embeddings(self, text_emb_dir: str) -> Tuple[np.ndarray, Dict[str, int]]:
        emb_path = os.path.join(text_emb_dir, "reports_emb.npy")
        map_path = os.path.join(text_emb_dir, "image_to_report_idx.json")
        if not os.path.exists(emb_path):
            raise FileNotFoundError(f"Missing report embeddings: {emb_path}")
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Missing image-to-report map: {map_path}")

        reports_emb = np.load(emb_path)
        with open(map_path, "r") as f:
            image_to_idx = json.load(f)
        if not isinstance(image_to_idx, dict):
            raise ValueError("image_to_report_idx.json must contain a dict mapping image stem to index")
        return reports_emb, {str(k): int(v) for k, v in image_to_idx.items()}

    def _load_metadata(self) -> Dict[str, Dict[str, str]]:
        split_file = self.config.dataset.get("split_file", None)
        if not split_file:
            split_file = os.path.join(self.root_folder, "qatacov_split.xlsx")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

        df = pd.read_excel(split_file)
        required = {"Image", "Split", "Report"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in {split_file}: {sorted(missing)}")

        metadata: Dict[str, Dict[str, str]] = {"train": {}, "val": {}, "test": {}}
        for _, row in df[["Image", "Split", "Report"]].dropna().iterrows():
            split = str(row["Split"]).strip().lower()
            image_name = str(row["Image"]).strip()
            if split in metadata:
                metadata[split][image_name] = str(row["Report"])
        return metadata

    def _get_ordered_images_path(self) -> Tuple[List, List, List, List, List, List]:
        images_dir = os.path.join(self.root_folder, "Images")
        masks_dir = os.path.join(self.root_folder, "Ground-truths")

        split_file = self.config.dataset.get("split_file", None)
        if not split_file:
            split_file = os.path.join(self.root_folder, "qatacov_split.xlsx")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

        df = pd.read_excel(split_file)
        required = {"Image", "Split"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in {split_file}: {sorted(missing)}")

        def build_paths(split: str) -> Tuple[List[str], List[str]]:
            split_df = df[df["Split"].astype(str).str.lower() == split]
            images = []
            labels = []
            for image_name in split_df["Image"].dropna().astype(str).tolist():
                image_path = os.path.join(images_dir, image_name)
                mask_name = f"mask_{image_name}"
                mask_path = os.path.join(masks_dir, mask_name)
                if os.path.exists(image_path) and os.path.exists(mask_path):
                    images.append(image_path)
                    labels.append(mask_path)
            return images, labels

        train_images, train_labels = build_paths("train")
        if self.validation:
            val_images, val_labels = build_paths("val")
        else:
            val_images, val_labels = [], []
        test_images, test_labels = build_paths("test")

        return train_images, train_labels, val_images, val_labels, test_images, test_labels

    def get_loader(self, split: str):
        assert split in ["train", "val", "test"], "Split must be train or val or test"

        if split == "train":
            return DataLoader(
                self.train_set,
                batch_size=self.config.dataset["batch_size"],
                shuffle=True,
                num_workers=self.NUM_WORKERS,
                pin_memory=False,
            )
        if split == "val":
            if self.validation:
                return DataLoader(
                    self.val_set,
                    batch_size=1,
                    num_workers=self.NUM_WORKERS,
                    pin_memory=False,
                )
            return
        return DataLoader(
            self.test_set,
            batch_size=1,
            num_workers=self.NUM_WORKERS,
            pin_memory=False,
        )


class QaTaCov2DTextEmbSet(Dataset):
    """
    Dataset returning image/mask pairs, report text and report embeddings for QaTaCov.
    """

    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        report_map: Optional[Dict[str, str]],
        reports_emb: np.ndarray,
        image_to_idx: Dict[str, int],
        transform=None,
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.report_map = report_map or {}
        self.reports_emb = reports_emb
        self.image_to_idx = image_to_idx
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
            "image": image_tensor,
            "label": gt,
            "image_path": image_path,
            "label_path": mask_path,
        }

        if self.report_map:
            image_name = os.path.basename(image_path)
            sample["text"] = self.report_map.get(image_name, "")

        image_stem = os.path.splitext(os.path.basename(image_path))[0]
        emb_idx = self.image_to_idx.get(image_stem)
        if emb_idx is None:
            raise KeyError(f"No text embedding index found for image stem: {image_stem}")
        sample["text_emb"] = torch.from_numpy(self.reports_emb[emb_idx]).float()
        sample["report_idx"] = torch.tensor(int(emb_idx), dtype=torch.long)

        return sample