import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
from base.base_trainer import BaseTrainer
from tqdm import tqdm
import torchio as tio
from utils.util import _onehot_enc_2d
import matplotlib.pyplot as plt
import numpy as np


class Trainer_2D2(BaseTrainer):
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

        for idx, sample in tqdm(enumerate(self.train_loader), desc=f'Epoch {epoch}', total=len(self.train_loader)):
            image = sample['image'].float().to(self.device)
            label = _onehot_enc_2d(sample['label'].long().to(self.device), self.num_classes).float().to(self.device)

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
        loader = getattr(self, f'{phase}_loader')
        metrics_manager = getattr(self, f'{phase}_metrics')

        for idx, sample in tqdm(enumerate(loader), desc=f'{phase}, epoch {epoch}', total=len(loader)):
            image = sample['image'].float().to(self.device)
            label = sample['label'].float().to(self.device)
            prediction = self.model(image)

            if idx == 0: 
                #save first batch predictions for visualization
                self._save_visualization(image, label, prediction, phase, epoch, idx)
                print("sample image path ", sample["image_path"][0])
                print("sample mask path ", sample["label_path"][0])

            loss = self.loss(prediction, label)

            metrics_manager.update_metrics(prediction, label)

        # After all iterations in the epoch, compute and store the epoch metrics
        metrics_manager.compute_epoch_metrics(epoch)
        #metrics_manager.log_to_wandb()
        metrics_manager.save_to_csv(self.save_path)

        results = self._results_dict(phase, epoch)

        return results

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
    
    def _save_visualization(self, image, label, prediction, phase, epoch, idx):
        """
        Save visualization of the first batch of predictions

        :param image: Input image tensor
        :param label: Ground truth label tensor
        :param prediction: Model prediction tensor
        :param phase: 'val' or 'test'
        :param epoch: Current epoch number
        :param idx: Current batch index
        """

        # Move tensors to CPU and convert to numpy
        image_np = image[0].cpu().numpy().squeeze()
        label_np = label[0].cpu().numpy().squeeze()
        prediction_np = prediction[0].cpu().numpy().squeeze()

        pred = torch.softmax(torch.tensor(prediction_np), dim=0)
        prediction_np = torch.argmax(pred, dim=0).numpy()



        # Create a figure with subplots
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].imshow(image_np, cmap='gray')
        axs[0].set_title('Input Image')
        axs[0].axis('off')

        axs[1].imshow(label_np, cmap='jet', alpha=0.5)
        axs[1].set_title('Ground Truth')
        axs[1].axis('off')

        axs[2].imshow(prediction_np, cmap='jet', alpha=0.5)
        axs[2].set_title('Prediction')
        axs[2].axis('off')

        plt.suptitle(f'{phase.capitalize()} Epoch {epoch} Visualization')
        plt.tight_layout()
        
        # Create directory if it doesn't exist
        vis_dir = os.path.join(self.save_path, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)

        # Save the figure
        plt.savefig(os.path.join(vis_dir, f'{phase}_epoch_{epoch}_visualization.png'))
        plt.close()