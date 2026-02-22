import torch
import torch.nn as nn
import monai

from .sigloss import SigLoss, DistributedSigLoss


class CombinedSegSigLoss(nn.Module):
    """
    L = seg_loss(pred, target) + lambda_sig * sig_loss(img_emb, txt_emb)

    - seg_loss_name: must exist in monai.losses
    - If img_emb/txt_emb are not provided, returns only seg_loss (baseline-compatible).
    - Supports linearly increasing lambda_sig from lambda_sig_initial to lambda_sig_final over epochs
    """
    def __init__(
        self,
        seg_loss_name: str = "DiceCELoss",
        seg_loss_kwargs: dict = None,
        lambda_sig: float = 0.1,
        lambda_sig_initial: float = None,
        lambda_sig_final: float = None,
        lambda_warmup_epochs: int = None,
        lambda_cooldown_epochs: int = None,
        sigloss_kwargs: dict = None
    ):
        super().__init__()

        seg_loss_kwargs = seg_loss_kwargs or {}
        sigloss_kwargs = sigloss_kwargs or {}

        if seg_loss_name not in monai.losses.__dict__:
            raise ValueError(f"seg_loss_name='{seg_loss_name}' not found in monai.losses")

        self.seg_loss = getattr(monai.losses, seg_loss_name)(**seg_loss_kwargs)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            distributed = True
        else:
            distributed = False
        if distributed:
            self.sig_loss = DistributedSigLoss(**sigloss_kwargs)
        else:
            self.sig_loss = SigLoss(**sigloss_kwargs)
        
        # Linear warmup and cooldown support
        if lambda_sig_initial is not None and lambda_sig_final is not None:
            self.lambda_sig_initial = float(lambda_sig_initial)
            self.lambda_sig_final = float(lambda_sig_final)
            self.lambda_warmup_epochs = lambda_warmup_epochs or 1
            self.lambda_cooldown_epochs = lambda_cooldown_epochs or 1
            self.use_warmup = True
        else:
            self.lambda_sig = float(lambda_sig)
            self.use_warmup = False
    
    def get_current_lambda(self, current_epoch: int = None):
        """
        Calculate current lambda_sig based on linear warmup and cooldown schedule.
        Phase 1 (0 to warmup_epochs): linear increase from initial to final
        Phase 2 (warmup_epochs to warmup_epochs + cooldown_epochs): linear decrease from final to 0
        Phase 3 (after cooldown): stays at 0
        """
        if not self.use_warmup:
            return self.lambda_sig
        
        if current_epoch is None:
            return self.lambda_sig_final
        
        # Phase 1: Warmup - linear increase
        if current_epoch < self.lambda_warmup_epochs:
            progress = current_epoch / self.lambda_warmup_epochs
            return self.lambda_sig_initial + (self.lambda_sig_final - self.lambda_sig_initial) * progress
        
        # Phase 2: Cooldown - linear decrease to 0
        cooldown_start = self.lambda_warmup_epochs
        cooldown_end = self.lambda_warmup_epochs + self.lambda_cooldown_epochs
        
        if current_epoch < cooldown_end:
            progress = (current_epoch - cooldown_start) / self.lambda_cooldown_epochs
            return self.lambda_sig_final * (1.0 - progress)
        
        # Phase 3: After cooldown - stay at 0
        return 0.0

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, img_emb=None, txt_emb=None, t_prime=None, bias=None, current_epoch=None, metric_manager=None):
        seg = self.seg_loss(prediction, target)

        # Get current lambda (with warmup if enabled)
        lambda_current = self.get_current_lambda(current_epoch)

        # If embeddings are missing, behave like baseline loss
        if img_emb is None or txt_emb is None or lambda_current == 0.0:
            loss_dict = {
                'seg_loss': seg,
                'sig_loss': torch.tensor(0.0, device=seg.device),
                'lambda_sig': lambda_current
            }
            #ADD TO WANDB, TO BE REMOVED
            if metric_manager is not None:
                metric_manager.metric_sums['DiceFocalLoss'] = metric_manager.metric_sums.get('DiceFocalLoss', 0.0) + seg.item()
                metric_manager.metric_counts['DiceFocalLoss'] = metric_manager.metric_counts.get('DiceFocalLoss', 0) + 1
                metric_manager.metric_sums['CombinedSegSigLoss'] = metric_manager.metric_sums.get('CombinedSegSigLoss', 0.0) + seg.item()
                metric_manager.metric_counts['CombinedSegSigLoss'] = metric_manager.metric_counts.get('CombinedSegSigLoss', 0) + 1
        
            return seg, loss_dict

        sig = self.sig_loss(img_emb, txt_emb, t_prime, bias)
        total_loss = seg + lambda_current * sig
        
        loss_dict = {
            'seg_loss': seg,
            'sig_loss': sig,
            'lambda_sig': lambda_current
        }


        #ADD TO WANDB, TO BE REMOVED
        if metric_manager is not None:
            metric_manager.metric_sums['Sigmoid_loss'] = metric_manager.metric_sums.get('Sigmoid_loss', 0.0) + sig.item()
            metric_manager.metric_counts['Sigmoid_loss'] = metric_manager.metric_counts.get('Sigmoid_loss', 0) + 1
            metric_manager.metric_sums['DiceFocalLoss'] = metric_manager.metric_sums.get('DiceFocalLoss', 0.0) + seg.item()
            metric_manager.metric_counts['DiceFocalLoss'] = metric_manager.metric_counts.get('DiceFocalLoss', 0) + 1
            metric_manager.metric_sums['CombinedSegSigLoss'] = metric_manager.metric_sums.get('CombinedSegSigLoss', 0.0) + total_loss.item()
            metric_manager.metric_counts['CombinedSegSigLoss'] = metric_manager.metric_counts.get('CombinedSegSigLoss', 0) + 1
        
        return total_loss, loss_dict
