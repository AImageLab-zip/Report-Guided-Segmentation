from typing import Tuple, List
import os
from base import BaseDataset3D
import random
import json

class ATLAS(BaseDataset3D):
    """
    Class for ATLAS 2.0 dataset creation and loading
    """
    def __init__(self, config, root_folder, validation=True, train_transforms=None, test_transforms=None):
        super().__init__(config, root_folder, validation, train_transforms, test_transforms)


    def _get_ordered_images_path(self) -> Tuple[List, List, List, List, List, List]:

        train_images_path = os.path.join(self.root_folder, 'imagesTr')
        train_labels_path = os.path.join(self.root_folder, 'labelsTr')
        test_images_path = os.path.join(self.root_folder, 'imagesTs')
        test_labels_path = os.path.join(self.root_folder, 'labelsTs')

        test_images = sorted(os.listdir(test_images_path))
        test_labels = sorted(os.listdir(test_labels_path))

        test_images = [os.path.join(test_images_path, filename) for filename in test_images]
        test_labels = [os.path.join(test_labels_path, filename) for filename in test_labels]

        _train_images = sorted(os.listdir(train_images_path))
        _train_labels = sorted(os.listdir(train_labels_path))

        if self.validation:
            # If a split file is specified, follows the specified train/val split
            if self.config.dataset.get("split_file"):
                split_file = self.config.dataset.get("split_file")
                f = json.load(open(split_file, 'r'))
                
                train_images = [i for i in f["train"] if i in _train_images]
                train_labels = [l for l in f["train"] if l in _train_labels]
                val_images = [i for i in f["val"] if i in _train_images]
                val_labels = [l for l in f["val"] if l in _train_labels]
            # If not, random train/val split with a seed for reproducibility
            else:
                random.seed(42)
                if self.config.dataset.get("val_ratio"):
                    val_percentage = self.config.dataset["val_ratio"]
                else:
                    val_percentage = 0.2

                num_val_samples = int(val_percentage * len(_train_images))
                val_idx = random.sample(range(len(_train_images)), num_val_samples)

                train_images = [img for idx, img in enumerate(_train_images) if idx not in val_idx]
                train_labels = [lbl for idx, lbl in enumerate(_train_labels) if idx not in val_idx]

                val_images = [_train_images[i] for i in val_idx]
                val_labels = [_train_labels[i] for i in val_idx]

            train_images = [os.path.join(train_images_path, filename) for filename in train_images]
            train_labels = [os.path.join(train_labels_path, filename) for filename in train_labels]
            val_images = [os.path.join(train_images_path, filename) for filename in val_images]
            val_labels = [os.path.join(train_labels_path, filename) for filename in val_labels]
        else:
            train_images = [os.path.join(train_images_path, filename) for filename in _train_images]
            train_labels = [os.path.join(train_labels_path, filename) for filename in _train_labels]
            val_images = []
            val_labels = []

        return train_images, train_labels, val_images, val_labels, test_images, test_labels


    def get_loader(self, split):

        assert split in ['train', 'val', 'test'], 'Split must be train or val or test'

        if split == 'train':
            return self._get_patch_loader(self.train_set)
        elif split == 'val':
            if self.validation:
                return self._get_volume_loader(self.val_set)
            else:
                return
        elif split == 'test':
            return self._get_volume_loader(self.test_set)

