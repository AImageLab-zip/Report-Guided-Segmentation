import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from torchvision.utils import make_grid
from base.base_trainer import BaseTrainer
from tqdm import tqdm
from collections import defaultdict
import os
from utils.util import _onehot_enc

class Trainer_2Dsliced(BaseTrainer):
    """
    Trainer class which implements a Basetrainer
    """
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()

        for idx, sample in tqdm(enumerate(self.train_loader), desc=f'Epoch {epoch}', total=len(self.train_loader)):
            image = sample['image'].to(self.device)
            label = _onehot_enc(sample['label'], self.num_classes).float().to(self.device)

            prediction = self.model(image)

            loss = self.loss(prediction, label)
            if self.debug:
                print(f"E: {epoch}\tI: {idx}\tL: {loss.item()}")

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            self.optimizer.step()

            self.train_metrics.update_metrics(prediction, label)

        # After all iterations in the epoch, compute and store the epoch metrics
        self.train_metrics.compute_epoch_metrics(epoch)
        #self.train_metrics.log_to_wandb()
        self.train_metrics.save_to_csv(self.save_path)
        results = self._results_dict('train', epoch)

        if self.debug:
            print(results)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return results

    @torch.inference_mode() #Context manager analogous to no_grad
    def eval_epoch(self, epoch, phase):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :param phase: val/test
        :return: A dictionary that contains information about validation/test metrics
        """
        assert phase in ['val', 'test'], f'phase should be val, or test, passed: {phase}'
        self.model.eval()
        loader = getattr(self, f'{phase}_loader')
        metrics_manager = getattr(self, f'{phase}_metrics')
        for idx, sample in tqdm(enumerate(loader), desc=f'{phase}, epoch {epoch}', total=len(loader)):
            image = sample['image'].to(self.device)
            label = _onehot_enc(sample['label'], self.num_classes).float().to(self.device)
            prediction = self.model(image)

            prediction = prediction.argmax(dim=1, keepdim=True).cpu()
            label = label.int().cpu()

            # controllare se va bene il calcolo delle metriche nonostante l'argmax?
            metrics_manager.update_metrics(prediction, label)

        # After all iterations in the epoch, compute and store the epoch metrics
        metrics_manager.compute_epoch_metrics(epoch)
        #metrics_manager.log_to_wandb()
        metrics_manager.save_to_csv()

        results = self._results_dict(phase, epoch)

        return results


    def _results_dict(self, phase, epoch):
        metrics_manager = getattr(self, f'{phase}_metrics')
        results = {self.loss_name: metrics_manager.get_metric_at_epoch(self.loss_name, epoch)}

        for m_name in metrics_manager.metrics.keys():
            if 'loss' not in m_name.lower():
                results[m_name] = self.val_metrics.get_metric_at_epoch(f'{m_name}_mean', epoch)

        return results

