import torch
import torch.nn.functional as F
from tqdm import tqdm

from base.base_trainer_text_guided import BaseTrainerText
from utils.util import _onehot_enc_2d


class Trainer_2D_TextGuided(BaseTrainerText):
    """
    2D trainer with (optional) report-guided training.

    Expected dataset fields (per sample / per batch):
        sample = {
            "image":      Tensor,                 # float in [0,1]
            "label":      Tensor,                 # float {0,1} or class ids depending on your pipeline
            "image_path": str,
            "label_path": str,
            "text_emb":   Tensor [768] or [B,768] # precomputed BioBERT pooled embeddings (offline)
        }
    """

    def _train_epoch(self, epoch):
        self.model.train()
        if self.use_text_guidance:
            self.guidance_head.train()

        if epoch == self.start_epoch:
            self._log_trainable_params_summary()

        for idx, sample in tqdm(
            enumerate(self.train_loader),
            desc=f"Epoch {epoch}",
            total=len(self.train_loader),
        ):
            image = sample["image"].float().to(self.device)
            label_raw = sample["label"].float().to(self.device)
            label = _onehot_enc_2d(label_raw, self.num_classes)

            if self.use_text_guidance:
                pred, bottleneck = self.model(image, return_bottleneck=True)

                if self.text_emb_key not in sample:
                    raise KeyError(
                        f"Missing '{self.text_emb_key}' in sample. "
                        f"Your dataset must return precomputed embeddings under this key."
                    )

                text_emb = sample[self.text_emb_key].to(self.device)  # [B,768] or [768]
                if text_emb.dim() == 1:
                    text_emb = text_emb.unsqueeze(0)

                # --- project both modalities into shared space ---
                # z_img: [B,d], z_txt: [B,d]
                z_img, z_txt = self.guidance_head(bottleneck, text_emb)

                # --- combined segsig loss ---
                loss = self.loss(pred, label, img_emb=z_img, txt_emb=z_txt,t_prime = self.model.t_prime,bias = model.bias)

            else:
                pred = self.model(image)
                loss = self.loss(pred, label)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if idx == 0:
                self._log_gradient_flow_summary()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            if self.use_text_guidance:
                torch.nn.utils.clip_grad_norm_(self.guidance_head.parameters(), 1.0)

            self.optimizer.step()

            self.train_metrics.update_metrics(pred, label)

        self.train_metrics.compute_epoch_metrics(epoch)
        self.train_metrics.save_to_csv(self.save_path)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return self._results_dict("train", epoch)


    def _log_trainable_params_summary(self):
        model_total = sum(p.numel() for p in self.model.parameters())
        model_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[Params] UNet total: {model_total:,} | trainable: {model_trainable:,}")

        if self.use_text_guidance and self.guidance_head is not None:
            head_total = sum(p.numel() for p in self.guidance_head.parameters())
            head_trainable = sum(p.numel() for p in self.guidance_head.parameters() if p.requires_grad)
            print(f"[Params] Guidance head total: {head_total:,} | trainable: {head_trainable:,}")


    def _log_gradient_flow_summary(self):
        model_grad_params = sum(1 for p in self.model.parameters() if p.requires_grad and p.grad is not None)
        model_zero_grads = sum(1 for p in self.model.parameters() if p.requires_grad and p.grad is not None and torch.allclose(p.grad, torch.zeros_like(p.grad)))
        print(f"[Grad] UNet params with grad: {model_grad_params} | zero grads: {model_zero_grads}")

        if self.use_text_guidance and self.guidance_head is not None:
            head_grad_params = sum(1 for p in self.guidance_head.parameters() if p.requires_grad and p.grad is not None)
            head_zero_grads = sum(1 for p in self.guidance_head.parameters() if p.requires_grad and p.grad is not None and torch.allclose(p.grad, torch.zeros_like(p.grad)))
            print(f"[Grad] Guidance head params with grad: {head_grad_params} | zero grads: {head_zero_grads}")


    @torch.inference_mode()
    def eval_epoch(self, epoch, phase):
        assert phase in ["val", "test"]
        self.model.eval()

        loader = getattr(self, f"{phase}_loader")
        metrics_manager = getattr(self, f"{phase}_metrics")

        for idx, sample in tqdm(
            enumerate(loader),
            desc=f"{phase}, epoch {epoch}",
            total=len(loader),
        ):
            image = sample["image"].float().to(self.device)

            label = sample["label"].long().to(self.device)

            pred = self.model(image)
            metrics_manager.update_metrics(pred, label)

        metrics_manager.compute_epoch_metrics(epoch)
        metrics_manager.save_to_csv(self.save_path)
        return self._results_dict(phase, epoch)


    def _results_dict(self, phase, epoch):
        metrics_manager = getattr(self, f"{phase}_metrics")

        if phase in ["train", "val"]:
            results = {self.loss_name: metrics_manager.get_metric_at_epoch(self.loss_name, epoch)}
        else:
            results = {}

        for m_name in metrics_manager.metrics.keys():
            if "loss" not in m_name.lower():
                metric_data = metrics_manager.get_metric_at_epoch(f"{m_name}_mean", epoch)
                results[m_name] = metric_data

                if f"{m_name}_aggregated_mean" in metrics_manager.data.columns:
                    aggregated_data = metrics_manager.get_metric_at_epoch(f"{m_name}_aggregated_mean", epoch)
                    results[m_name].update(aggregated_data)

        return results
