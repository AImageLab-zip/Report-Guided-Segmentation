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
import itertools
from utils.util import _onehot_enc
from datasets.DatasetFactory import DatasetFactory
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class Trainer_3DText(BaseTrainer):
    """
    Trainer class which implements a Basetrainer
    """
    def __init__(self, *args, pretrained_path=None, **kwargs):
        self.pretrained_path = pretrained_path
        super().__init__(*args, **kwargs)
        if self.pretrained_path:
            self._load_pretrained_weights()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_sampler.set_epoch(epoch)
        self.train_reports_sampler.set_epoch(epoch) if self.train_reports_sampler is not None else None
        iterator = tqdm(enumerate(self.train_loader), desc=f'Epoch {epoch}', total=len(self.train_loader)) if (dist.get_rank()==0 and not self.debug) else enumerate(self.train_loader)

        def _train_step(sample, use_report=False):
            image = sample['image'][tio.DATA].float().to(self.device)
            label = _onehot_enc(sample['label'][tio.DATA].long(), self.num_classes).float().to(self.device)

            report = None
            if use_report and 'report' in sample:
                report = sample['report'].to(self.device)

            with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                if report is not None:
                    prediction, z_img, z_txt = self.model(image, report)
                    #if isinstance(prediction, (tuple, list)):
                        #prediction = prediction[0]

                    loss = self.loss(prediction, label, z_img, z_txt, self.model.module.t_prime, self.model.module.bias)
                else:
                    prediction = self.model(image)
                    loss = self.loss(prediction, label)

            self.optimizer.zero_grad(set_to_none=True)
            if self.use_scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.) # better 5 with SGD, 1 with Adam
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.) # better 5 with SGD, 1 with Adam
                self.optimizer.step()

            self.train_metrics.update_metrics(prediction, label)
            return loss

        report_every_k = 2
        reports_iter = None
        if hasattr(self, 'train_reports_loader') and self.train_reports_loader is not None:
            if len(self.train_reports_loader) > 0:
                reports_iter = itertools.cycle(self.train_reports_loader)

        for idx, sample in iterator:
            loss = _train_step(sample, use_report=False)
            if self.debug:
                print(f"E: {epoch}\tI: {idx}\tL: {loss.item()}")

            if reports_iter is not None and (idx + 1) % report_every_k == 0:
                report_sample = next(reports_iter)
                _train_step(report_sample, use_report=True)

        # After all iterations in the epoch, compute and store the epoch metrics
        self.train_metrics.compute_epoch_metrics(epoch)
   
        self.train_metrics.save_to_csv(self.save_path)
        results = self._results_dict('train', epoch)

        if self.debug:
            print(results)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        torch.cuda.empty_cache()
        dist.barrier()
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
        iterator = tqdm(enumerate(loader), desc=f'{phase}, epoch {epoch}', total=len(loader)) if (dist.get_rank() == 0 and not self.debug) else enumerate(loader)

        for idx, sample in iterator:
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
            del prediction, label

        # After all iterations in the epoch, compute and store the epoch metrics
        metrics_manager.compute_epoch_metrics(epoch)
        metrics_manager.save_to_csv(self.save_path)

        results = self._results_dict(phase, epoch)
        torch.cuda.empty_cache()
        dist.barrier()
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

        #num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))

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
    

    def _init_dataset(self, fold=0, split_ratio = (1, 0)):
        """
        Override the _init_dataset method to initialize also the train dataloader with samples with text

        :param fold: number of fold in case of k-fold cross-validation
        :param split_ratio: current epoch number
        """
        self.dataset = DatasetFactory.create_instance(self.config, self.validation, self.train_transforms, self.test_transforms)
        self.train_loader, self.train_sampler = self.dataset.get_loader('train')
        self.test_loader = self.dataset.get_loader('test')
        self.train_reports_loader, self.train_reports_sampler = self.dataset.get_loader('train', report=True)
        if self.validation:
            self.val_loader = self.dataset.get_loader('val')
        else:
            self.val_loader = None 

    def _load_pretrained_weights(self):
        if not self.pretrained_path:
            return

        if not os.path.exists(self.pretrained_path):
            raise FileNotFoundError(f"pretrained_unet_path not found: {self.pretrained_path}")

        ckpt = torch.load(self.pretrained_path, map_location=self.device)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model_to_load = self.model.module if isinstance(self.model, DDP) else self.model
        missing, unexpected = model_to_load.load_state_dict(state_dict, strict=False)
        print(f"[Pretrain] Loaded model weights from {self.pretrained_path}")
        print(f"[Pretrain] Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
        print(f"[Pretrain] Missing keys: {missing} | Unexpected keys: {unexpected}")
        


