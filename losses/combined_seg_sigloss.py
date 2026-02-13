import torch
import torch.nn as nn
import monai

from .sigloss import SigLoss, DistributedSigLoss


class CombinedSegSigLoss(nn.Module):
    """
    L = seg_loss(pred, target) + lambda_sig * sig_loss(img_emb, txt_emb)

    - seg_loss_name: must exist in monai.losses
    - If img_emb/txt_emb are not provided, returns only seg_loss (baseline-compatible).
    """
    def __init__(
        self,
        seg_loss_name: str = "DiceCELoss",
        seg_loss_kwargs: dict = None,
        lambda_sig: float = 0.1,
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
        self.lambda_sig = float(lambda_sig)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, img_emb=None, txt_emb=None, t_prime=None, bias=None):
        seg = self.seg_loss(prediction, target)

        # If embeddings are missing, behave like baseline loss
        if img_emb is None or txt_emb is None or self.lambda_sig == 0.0:
            return seg

        sig = self.sig_loss(img_emb, txt_emb, t_prime, bias)
        return seg + self.lambda_sig * sig
