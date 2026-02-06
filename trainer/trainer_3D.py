import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from torch.cuda.amp import autocast
from torchvision.utils import make_grid
from base.base_trainer import BaseTrainer
from tqdm import tqdm
import torchio as tio
from collections import defaultdict
import os
from utils.util import _onehot_enc

class Trainer_3D(BaseTrainer):
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
            image = sample['image'][tio.DATA].float().to(self.device)
            label = _onehot_enc(sample['label'][tio.DATA].long(), self.num_classes).float().to(self.device)

            with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                prediction = self.model(image)
                loss = self.loss(prediction, label)
            if self.debug:
                print(f"E: {epoch}\tI: {idx}\tL: {loss.item()}")

            self.optimizer.zero_grad(set_to_none=True)
            if self.use_scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.) # better 5 with SGD, 1 with Adam
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.) # better 5 with SGD, 1 with Adam
                self.optimizer.step()

            self.train_metrics.update_metrics(prediction, label)

        # After all iterations in the epoch, compute and store the epoch metrics
        self.train_metrics.compute_epoch_metrics(epoch)
   
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
            # Convert batch dictionary back to Subject (batch_size=1)
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=sample['image']['data'][0]),
                label=tio.LabelMap(tensor=sample['label']['data'][0])
            )
            loader_patches, pred_aggregator, label_aggregator = self._inference_sampler(subject)

            #Loop over the patches
            for j, patch in enumerate(loader_patches):
                image = patch['image'][tio.DATA].float().to(self.device)
                label = patch['label'][tio.DATA].float().to(self.device)
                prediction = self.model(image)
                pred_aggregator.add_batch(prediction, patch[tio.LOCATION])
                label_aggregator.add_batch(label, patch[tio.LOCATION])

            prediction = pred_aggregator.get_output_tensor().unsqueeze(0).cpu()
            label = label_aggregator.get_output_tensor().unsqueeze(0).int().cpu()

            # Pass raw predictions (logits) to metrics - they handle conversion as needed
            metrics_manager.update_metrics(prediction, label)

        # After all iterations in the epoch, compute and store the epoch metrics
        metrics_manager.compute_epoch_metrics(epoch)
        metrics_manager.save_to_csv(self.save_path)

        results = self._results_dict(phase, epoch)

        return results

    def _inference_sampler(self, sample: tio.Subject):

        patch_size_value = self.config.dataset['patch_size']
        if isinstance(patch_size_value, int):
            patch_size = (patch_size_value, patch_size_value, patch_size_value)
        else:
            patch_size = tuple(patch_size_value)
        image_shape = sample.spatial_shape

        if any(p > s for p, s in zip(patch_size, image_shape)):
            sample = tio.CropOrPad(patch_size)(sample)

        # Grid samplers are useful to perform inference using all patches from a volume
        grid_sampler = tio.data.GridSampler(
            sample,
            patch_size,
            self.config.dataset['grid_overlap']
        )

        # Aggregate patches for dense inference
        pred_aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode="hann")
        label_aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode="hann")

        num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))

        loader = tio.SubjectsLoader(
            grid_sampler,
            #num_workers=num_workers,
            num_workers=0, #to not have the warning 
            batch_size=1,
            pin_memory=False, #true
        )

        return loader, pred_aggregator, label_aggregator

    def _results_dict(self, phase, epoch):
        metrics_manager = getattr(self, f'{phase}_metrics')
        if phase in ['train', 'val']:
            results = {self.loss_name: metrics_manager.get_metric_at_epoch(self.loss_name, epoch)}
        else:
            results = {}

        for m_name in metrics_manager.metrics.keys():
            if 'loss' not in m_name.lower():
                metric_data = metrics_manager.get_metric_at_epoch(f'{m_name}_mean', epoch)
                results[m_name] = metric_data
                
                # Also include aggregated_mean if it exists
                if f'{m_name}_aggregated_mean' in metrics_manager.data.columns:
                    aggregated_data = metrics_manager.get_metric_at_epoch(f'{m_name}_aggregated_mean', epoch)
                    results[m_name].update(aggregated_data)

        return results

