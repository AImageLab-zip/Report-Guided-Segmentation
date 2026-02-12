import os
import torchio as tio
import nibabel as nib
from torch.utils.data import DataLoader
from typing import List, Tuple
from torch.utils.data import DistributedSampler
import torch.distributed as dist
import torch
# TODO: k-fold cross validation
class BaseDataset:

    if 'SLURM_CPUS_PER_TASK' in os.environ:
        cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
        local_world = int(os.environ.get("LOCAL_WORLD_SIZE", 1))  # torchrun sets this
        NUM_WORKERS = max(1, cpus // local_world)
    else:
        NUM_WORKERS = 4  # Set to a fixed number if the environment variable does not exist

    def __init__(self, config, root_folder, validation=True, train_transforms=None, test_transforms=None, **kwargs):
        if dist.get_rank() == 0:
            print(f'Detected {self.NUM_WORKERS} cpus. rank{dist.get_rank()}')
        self.config = config
        self.validation = validation
        self.root_folder = root_folder
        self.train_images, self.train_labels, self.val_images, self.val_labels, self.test_images, self.test_labels = self._get_ordered_images_path()
        assert (len(self.train_images) == len(self.train_labels) and
                len(self.val_images) == len(self.val_labels) and
                len(self.test_images) == len(self.test_labels)), "Mismatch in data lengths"

        self.train_set = tio.SubjectsDataset(self._get_subjects_list('train'), transform=train_transforms)
        if self.validation:
            self.val_set = tio.SubjectsDataset(self._get_subjects_list('val'), transform=test_transforms)
        self.test_set = tio.SubjectsDataset(self._get_subjects_list('test'), transform=test_transforms)


    def _get_ordered_images_path(self,) -> Tuple[List, List, List, List, List, List]:
        """
        Get the ordered images and their paths.

        :return: A tuple containing six lists, which corresponds to images and labels lists respectively of train,
                validation (if required) and test set. validation lists can be None
        """
        raise NotImplementedError

    def _get_subjects_list(self, split : str) -> List[tio.Subject]:

        # Use getattr to dynamically access the attribute based on the split parameter
        images = getattr(self, f'{split}_images', [])
        labels = getattr(self, f'{split}_labels', [])
        subjects = []
        for image_path, label_path in zip(images, labels):
            # this is the dictionary returned by the dataloader through the iterations
            subject_dir = {
                'image_path': image_path,
                'label_path': label_path,
                'image': tio.ScalarImage(image_path),
                'label': tio.LabelMap(label_path)
            }
            subjects.append(tio.Subject(**subject_dir))
        print(f"Loaded {len(subjects)} subject for split {split}")
        return subjects

    # return a patch-based SubjectsLoader (dataloader of TorchIO)
    def _get_patch_loader(self, dataset: tio.SubjectsDataset, batch_size: int = 1,shuffle=True):

        # if you need a Weighted Sampler, implement it in the specific dataset (you will need a sampling map for each subject)
        sampler = tio.UniformSampler(
            patch_size=self.config.dataset['patch_size']
        )

        subjects_sampler = torch.utils.data.DistributedSampler(dataset, shuffle=shuffle)
        queue = tio.Queue(
                subjects_dataset=dataset,
                max_length=self.config.dataset['queue_length'],
                samples_per_volume=self.config.dataset['samples_per_volume'],
                sampler=sampler,
                subject_sampler= subjects_sampler,
                num_workers=self.NUM_WORKERS,
                shuffle_subjects=False,
                shuffle_patches=True,
                start_background=False,
        )
        # Use SubjectsLoader instaead of Torch Dataloader for PyTorch 2.3+ compatibility with TorchIO
        loader = tio.SubjectsLoader(
            queue,
            batch_size=batch_size,
            num_workers=0,  # Queue already handles multiprocessing
            pin_memory=True
        )
        return loader, subjects_sampler

    # return a SubjectsLoader (dataloader of TorchIO) which use entire volumes/image
    def _get_entire_loader(self, dataset: tio.SubjectsDataset, batch_size: int = 1,shuffle=False):
        subjects_sampler = torch.utils.data.DistributedSampler(dataset, shuffle=shuffle)
        loader = tio.SubjectsLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.NUM_WORKERS,
            pin_memory=True,
            shuffle=False,
            sampler=subjects_sampler
        )
        return loader

    def get_loader(self, split):
        """
        Get the correct Dataloader based on the phase (train, validation, test).

        :param split:
        :return: One of the Dataloader implemented above
        """
        raise NotImplementedError
