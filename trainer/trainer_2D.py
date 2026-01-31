import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
from base.base_trainer_2d import BaseTrainer2D
from tqdm import tqdm
import torchio as tio
from utils.util import _onehot_enc_2d


class Trainer_2D(BaseTrainer2D):
    """
    Trainer class which implements a Basetrainer for full 2D images.
    """

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        epoch_loss = 0.0
        n_batches = 0

        for idx, sample in tqdm(enumerate(self.train_loader), desc=f'Epoch {epoch}', total=len(self.train_loader)):
            image = sample['image'].float().to(self.device)
            label_raw = sample['label'].float().to(self.device)
            label = _onehot_enc_2d(label_raw)

            prediction = self.model(image)

            print("Label shape", label.shape)
            print("Prediction shape:", prediction.shape)

            loss = self.loss(prediction, label)
            if self.debug:
                print(f"E: {epoch}\tI: {idx}\tL: {loss.item()}")

            epoch_loss += loss.item()
            n_batches += 1

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            self.optimizer.step()

            self.train_metrics.update_metrics(prediction, label)

        # After all iterations in the epoch, compute and store the epoch metrics
        self.train_metrics.compute_epoch_metrics(epoch)
        #self.train_metrics.log_to_wandb()
        self.train_metrics.save_to_csv(self.save_path)

        mean_loss = epoch_loss / max(1, n_batches)
        results = self._results_dict('train', epoch, mean_loss=mean_loss)

        if self.debug:
            print(results)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return results

    @torch.inference_mode()
    def eval_epoch(self, epoch, phase):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :param phase: val/test
        :return: A dictionary that contains information about validation/test metrics
        """
        assert phase in ['val', 'test'], f'phase should be val, or test, passed: {phase}'
        self.model.eval()
        epoch_loss = 0.0
        n_batches = 0
        loader = getattr(self, f'{phase}_loader')
        metrics_manager = getattr(self, f'{phase}_metrics')

        for idx, sample in tqdm(enumerate(loader), desc=f'{phase}, epoch {epoch}', total=len(loader)):
            image = sample['image'].float().to(self.device)
            label = sample['label'].long().to(self.device)
            prediction = self.model(image)

            loss = self.loss(prediction, label)

            epoch_loss += loss.item()
            n_batches += 1

            metrics_manager.update_metrics(prediction, label)

        # After all iterations in the epoch, compute and store the epoch metrics
        metrics_manager.compute_epoch_metrics(epoch)
        #metrics_manager.log_to_wandb()
        metrics_manager.save_to_csv(self.save_path)

        mean_loss = epoch_loss / max(1, n_batches)

        results = self._results_dict(phase, epoch, mean_loss=mean_loss)

        return results

    def _results_dict(self, phase, epoch, mean_loss=None):
        metrics_manager = getattr(self, f'{phase}_metrics')
        results = {}
        # Only include loss when provided (train/val)
        if mean_loss is not None:
            results[self.loss_name] = {self.loss_name: float(mean_loss)}  # keep your dict-of-dicts style

        for m_name in metrics_manager.metrics.keys():
            metric_data = metrics_manager.get_metric_at_epoch(f'{m_name}_mean', epoch)
            results[m_name] = metric_data
        return results