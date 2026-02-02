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
from monai.transforms import Compose as MonaiCompose

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
    def __init__(self, config, epochs, validation, save_path, resume=False, debug=False, eval_metric_type='mean', use_wandb=False, save_visualizations = False, **kwargs):
        """
        Initialize the Trainer with model, optimizer, scheduler, loss, metrics and weights using the config file
        """
        self.config = config
        self.debug = debug
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.validation = validation
        self.eval_metric_type = eval_metric_type
        self.use_wandb = use_wandb
        self.wandb_run_id = None
        self.save_visualizations = save_visualizations

        self.model = ModelFactory.create_instance(self.config).to(self.device)

        self.optimizer, self.lr_scheduler = OptimizerFactory.create_instance(self.model, self.config)

        self.loss_name, self.loss = LossFactory.create_instance(self.config)

        # Metrics filtered by phase flags in config (missing flag => False)
        train_metrics_dict = MetricsFactory.create_instance(self.config, phase="train")
        val_metrics_dict = MetricsFactory.create_instance(self.config, phase="val") if self.validation else {}
        test_metrics_dict = MetricsFactory.create_instance(self.config, phase="test")

        # Add loss to train/val managers
        train_metrics_dict[self.loss_name] = self.loss
        if self.validation:
            val_metrics_dict[self.loss_name] = self.loss

        self.train_metrics = MetricsManager(self.config, "train", **train_metrics_dict)
        if self.validation:
            self.val_metrics = MetricsManager(self.config, "val", **val_metrics_dict)
        self.test_metrics = MetricsManager(self.config, "test", **test_metrics_dict)

        self.start_epoch = 1
        self.epochs = epochs
        self.save_period = 1
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.resume = resume
        self.eval_metric = self.config.metrics[0]['key'] # Can be parametrized based on the metric I would choose to save best_model (now the first one is used)
        self.best_metric = 0
        self.num_classes = config.model['num_classes']

        self._init_transforms()
        self._init_dataset()

        if self.use_wandb and not self.resume:
            self._init_wandb()

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

        backend = transforms_config.get("backend", "torchio")
        default_keys = tuple(transforms_config.get("keys", ["image", "label"]))

        preprocessing_transforms = TransformsFactory.create_instance(transforms_config.get('preprocessing', []), backend=backend, default_keys=default_keys)
        augmentation_transforms = TransformsFactory.create_instance(transforms_config.get('augmentations', []), backend=backend, default_keys=default_keys)

        # Compose the final transforms
        # For training: preprocessing + augmentations
        if preprocessing_transforms and augmentation_transforms:
            if backend == "torchio":
                self.train_transforms = tio.Compose([preprocessing_transforms, augmentation_transforms])
            else:
                self.train_transforms = MonaiCompose([preprocessing_transforms, augmentation_transforms])
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

    def _init_wandb(self, resume_id=None):
        """
        Initialize Weights & Biases logging.
        
        :param resume_id: Optional run ID to resume an existing W&B run
        """
        if not self.use_wandb:
            return

        wandb_project = os.environ.get('WANDB_PROJECT', 'medical-segmentation')
        wandb_entity = os.environ.get('WANDB_ENTITY', None)
        config_dict = {k: v for k, v in self.config.__dict__.items() if not k.startswith('_')}
        
        if resume_id:
            # Resume existing run
            # Note: When resuming, the project name must match the original run's project
            # W&B will validate this automatically and raise an error if there's a mismatch
            wandb.init(
                entity=wandb_entity,
                project=wandb_project,
                name=self.config.name,
                id=resume_id,
                resume="must",
                config=config_dict
            )
            print(f"Resumed W&B run: {self.config.name} (ID: {resume_id})")
            print(f"Entity: {wandb_entity}, Project: {wandb_project}")
        else:
            wandb.init(
                entity=wandb_entity,
                project=wandb_project,
                name=self.config.name,
                config=config_dict
            )
            self.wandb_run_id = wandb.run.id
            print(f"Started W&B run: {self.config.name} (ID: {self.wandb_run_id})")
            print(f"Entity: {wandb_entity}, Project: {wandb_project}")

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
            'wandb_run_id': self.wandb_run_id if self.use_wandb else None,
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

        if self.use_wandb:
            self.wandb_run_id = checkpoint.get('wandb_run_id', None)
            if self.wandb_run_id:
                self._init_wandb(resume_id=self.wandb_run_id)
            else:
                print("Warning: No W&B run ID found in checkpoint. Starting new run.")
                self._init_wandb()

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
            
            if self.use_wandb:
                self.train_metrics.log_to_wandb(epoch)
            
            save_best = False
            if epoch % self.save_period == 0:
                if self.validation:
                    _val_metrics = self.eval_epoch(epoch, 'val')
                    if self.use_wandb:
                        self.val_metrics.log_to_wandb(epoch)
                    
                    #val_metric = _val_metrics[list(self.metrics.keys())[0]]
                    metric_key = f'{self.eval_metric}_{self.eval_metric_type}'
                    eval_metric_value = _val_metrics[self.eval_metric][metric_key]
                    if eval_metric_value > self.best_metric:
                        self.best_metric = eval_metric_value
                        save_best = True
                        print(f'New best {metric_key}: {self.best_metric:.4f}')

            self._save_checkpoint(epoch, save_best=save_best)

            if epoch == self.epochs:
                self.eval_epoch(epoch, 'test')
                if self.use_wandb:
                    self.test_metrics.log_to_wandb(epoch)
        
        if self.use_wandb:
            wandb.finish()
            print("W&B run finished.")