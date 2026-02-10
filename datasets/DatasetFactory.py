import datasets
from base.base_dataset import BaseDataset
from base.base_dataset2d_sliced import BaseDataset2DSliced
from datasets import *
class DatasetFactory:
    @staticmethod
    def create_instance(config, validation, train_transforms=None, test_transforms=None) -> BaseDataset | BaseDataset2DSliced :
        dataset_name = config.dataset['name']
        root_folder = config.dataset['path']
        # TODO: Inizializzare qua le Transforms dato che sono ricavabili dal config?
        if dataset_name not in datasets.__dict__:
            raise Exception(f"Could not find dataset: {dataset_name}")
        dataset_class = getattr(datasets, dataset_name)

        try:
            # instantiate the dataset
            dataset = dataset_class(config, root_folder, validation, train_transforms, test_transforms)
        except TypeError as e:
            raise TypeError(f"Could not instantiate {dataset_name}\n{e}")

        return dataset