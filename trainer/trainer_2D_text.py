import torch
import torch.nn.functional as F
from tqdm import tqdm

from base.base_trainer_text import BaseTrainerText
from utils.util import _onehot_enc_2d


class Trainer_2D_Text(BaseTrainerText):
    """
    2D trainer with text guidance (text embeddings are precomputed offline).
    """

    def _train_epoch(self, epoch):
        self.model.train()
        if self.use_text_guidance:
            self.text_proj.train()
            if self.text_ln is not None:
                self.text_ln.train()

        for idx, sample in tqdm(enumerate(self.train_loader), desc=f'Epoch {epoch}', total=len(self.train_loader)):
            image = sample['image'].float().to(self.device)
            label_raw = sample['label'].float().to(self.device)
            label = _onehot_enc_2d(label_raw, self.num_classes)

            pred, img_emb = self.model(image, return_embedding=True)

            if self.use_text_guidance:
                if self.text_emb_key not in sample:
                    raise KeyError(f"Missing '{self.text_emb_key}' in sample. "
                                   f"Your dataset must return precomputed embeddings under this key.")
                text_emb = sample[self.text_emb_key].to(self.device)
                if text_emb.dim() == 1:
                    text_emb = text_emb.unsqueeze(0)
                text_emb_proj = self.project_text(text_emb)  # [B,1024]

                loss = self.loss(pred, label, img_emb=img_emb, txt_emb=text_emb_proj)

            else:
                loss = self.loss(pred, label)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            if self.use_text_guidance:
                torch.nn.utils.clip_grad_norm_(self.text_proj.parameters(), 1.)
                if self.text_ln is not None:
                    torch.nn.utils.clip_grad_norm_(self.text_ln.parameters(), 1.)
            self.optimizer.step()

            self.train_metrics.update_metrics(pred, label)

        self.train_metrics.compute_epoch_metrics(epoch)
        self.train_metrics.save_to_csv(self.save_path)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return self._results_dict('train', epoch)

    @torch.inference_mode()
    def eval_epoch(self, epoch, phase):
        assert phase in ['val', 'test']
        self.model.eval()
        loader = getattr(self, f'{phase}_loader')
        metrics_manager = getattr(self, f'{phase}_metrics')

        for idx, sample in tqdm(enumerate(loader), desc=f'{phase}, epoch {epoch}', total=len(loader)):
            image = sample['image'].float().to(self.device)
            label = sample['label'].long().to(self.device)
            pred = self.model(image)  # no embedding needed at inference
            metrics_manager.update_metrics(pred, label)

        metrics_manager.compute_epoch_metrics(epoch)
        metrics_manager.save_to_csv(self.save_path)
        return self._results_dict(phase, epoch)
    
    
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

                if f'{m_name}_aggregated_mean' in metrics_manager.data.columns:
                    aggregated_data = metrics_manager.get_metric_at_epoch(f'{m_name}_aggregated_mean', epoch)
                    results[m_name].update(aggregated_data)

        return results
