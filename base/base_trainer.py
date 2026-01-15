import torch
from abc import abstractmethod
from numpy import inf

import os
import yaml
from tqdm import tqdm
import torch
import torchio as tio
import numpy as np
import wandb
from collections import defaultdict

from torch.utils.data import DataLoader

import json
from models import ModelFactory
from losses import LossFactory
from optimizers import OptimizerFactory
from datasets import DatasetFactory
from metrics import MetricsFactory, MetricsManager
from transforms import TransformsFactory


class BaseTrainer:
    """
    Base class for all trainers
    """
    # Mandatory parameters not specified in the config file, must be passed as CL params when calling the main.py
    # If too many params it is possible to specify them in another file
    def __init__(self, config, epochs, validation, save_path, resume=False, debug=False, **kwargs):
        """
        Initialize the Trainer with model, optimizer, scheduler, loss, metrics and weights using the config file
        """
        self.config = config
        self.debug = debug
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.validation = validation

        self.model = ModelFactory.create_instance(self.config).to(self.device)

        self.optimizer, self.lr_scheduler = OptimizerFactory.create_instance(self.model, self.config)

        self.loss_name, self.loss = LossFactory.create_instance(self.config)
        self.metrics = MetricsFactory.create_instance(self.config)
        self.test_metrics = MetricsManager(self.config,'test', **self.metrics)
        #Append also the loss which has to be computed during the training and validation steps
        self.metrics[self.loss_name]=self.loss
        self.train_metrics = MetricsManager(self.config, 'train', **self.metrics)
        if self.validation:
            self.val_metrics = MetricsManager(self.config, 'val', **self.metrics)

        self.start_epoch = 1
        self.epochs = epochs
        self.save_period = 1
        self.save_path = save_path
        self.resume = resume
        self.best_metric = 0
        self.num_classes = config.model['num_classes']

        self._init_transforms()
        self._init_dataset()

        # Handle checkpoint resuming
        if self.resume:
            self._resume_checkpoint()

    def _init_transforms(self):
        """
        Initialize the preprocessing and augmentation transforms from the external configuration file using TransformsFactory.
        """
        transforms_path = self.config.dataset.get('transforms', None)
        if not transforms_path or not os.path.isfile(transforms_path):
            raise FileNotFoundError(f"Transforms file not found at path: {transforms_path}")

        with open(transforms_path, 'r') as f:
            transforms_config = json.load(f)

        preprocessing_transforms = TransformsFactory.create_instance(transforms_config.get('preprocessing', []))
        augmentation_transforms = TransformsFactory.create_instance(transforms_config.get('augmentations', []))

        # Compose the final transforms
        # For training: preprocessing + augmentations
        if preprocessing_transforms and augmentation_transforms:
            self.train_transforms = tio.Compose([preprocessing_transforms, augmentation_transforms])
            self.test_transforms = preprocessing_transforms
        elif preprocessing_transforms:
            self.train_transforms = preprocessing_transforms
            self.test_transforms = preprocessing_transforms
        elif augmentation_transforms:
            self.train_transforms = augmentation_transforms
            self.test_transforms = None
        else:
            self.train_transforms = None
            self.test_transforms = None

    # TODO: Gestire k-fold cross-val
    def _init_dataset(self, fold=0, split_ratio = (1, 0)):
        """
        Initializing the sets of the Dataset

        :param fold: number of fold in case of k-fold cross-validation
        :param split_ratio: current epoch number
        """
        self.dataset = DatasetFactory.create_instance(self.config, self.validation, self.train_transforms, self.test_transforms)
        self.train_loader = self.dataset.get_loader('train')
        self.test_loader = self.dataset.get_loader('test')
        if self.validation:
            self.val_loader = self.dataset.get_loader('val')
        else:
            self.val_loader = None

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    @abstractmethod
    def eval_epoch(self, epoch, phase):
        """
        Validation/Test step
        """
        raise NotImplementedError

    def _save_checkpoint(self, epoch, save_best):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param save_best: if True, save the checkpoint also to 'model_best.pth'
        """
        # TODO: Se può essere utile altro aggiungere qua
        state = {
            'name': type(self.model).__name__,
            'config': self.config,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            'epoch': epoch,
            'best_metric': self.best_metric,
        }

        checkpoint_path = os.path.join(self.save_path, 'model_last.pth')
        torch.save(state, checkpoint_path)

        if save_best:
            checkpoint_path = os.path.join(self.save_path, 'model_best.pth')
            print(f'Saving checkpoints {checkpoint_path}')
            torch.save(state, checkpoint_path)

    def _resume_checkpoint(self):
        """
        Resume from saved checkpoints
        """

        checkpoint_path = os.path.join(self.save_path, 'model_last.pth')
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint['model'])
        print("Model weights loaded.")

        # Load optimizer state
        if 'optimizer' in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Optimizer state loaded.")

        # Load learning rate scheduler state
        if 'lr_scheduler' in checkpoint and self.lr_scheduler is not None and checkpoint['lr_scheduler'] is not None:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            print("Learning rate scheduler state loaded.")

        # Set start epoch
        self.start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
        print(f"Resuming training from epoch {self.start_epoch}.")

        # Load best metric if available
        self.best_metric = checkpoint.get('best_metric', 0)
        print(f"Best metric so far: {self.best_metric}")

        # Load metrics data if available
        self.train_metrics.load_from_csv(self.save_path)
        self.test_metrics.load_from_csv(self.save_path)
        if self.validation:
            self.val_metrics.load_from_csv(self.save_path)

        # TODO: Check the meaning of this part and if it is required
        '''
        # Handle W&B resuming
        if wandb.run is not None:
            # If already initialized, resume the run
            wandb.config.update({'resume': True, 'resume_epoch': self.start_epoch}, allow_val_change=True)
            print("Resumed W&B run.")
        else:
            # Initialize W&B with resume flag
            wandb.init(project=self.config.name, resume='allow')
            wandb.config.update({'resume': True, 'resume_epoch': self.start_epoch})
            print("Initialized W&B run with resume.")
        '''

    def train(self):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            print(f'EPOCH {epoch}')
            epoch_results = self._train_epoch(epoch)
            #aggiorna best metric

            if self.debug:
                print(epoch_results)

            if epoch % self.save_period == 0:
                if self.validation:
                    _val_metrics = self.eval_epoch(epoch, 'val')
                    val_metric = _val_metrics[list(self.metrics.keys())[0]] #parametrizzare sulla base di cosa voglio salvare
                    if val_metric['DiceMetric_mean'] > self.best_metric: #parametrizzare sulla base di cosa voglio salvare
                        self.best_metric = val_metric
                        self._save_checkpoint(epoch, save_best=True)

            self._save_checkpoint(epoch, save_best=False)

            if epoch == self.epochs:
                self.eval_epoch(epoch, 'test')