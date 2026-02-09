import torch
import torch.nn as nn
import torch.nn.functional as F


class SigLoss(nn.Module):
    """
    Sigmoid contrastive loss (SigLIP-style) over a batch.

    Expects:
      img_emb: [B, D]
      txt_emb: [B, D]
    Returns:
      scalar loss

    Usage:
    criterion = SigLoss(mode = SigLoss.STANDARD)
    ...
    loss = criterion(img_emb, txt_emg)
    """
    STANDARD = 0        # Default SigLipLoss implementation
    ONLY_POSITIVES = 1  # Only positive terms contribute
    CONTINUOUS = 2      # Continuous weights in range [-1; +1] instead of hard -1 +1

    def __init__(
        self,
        temperature_init: float = 0.1,      
        bias_init: float = -10.0,             
        learnable_temperature: bool = True,
        learnable_bias: bool = True,
        mode = None
    ):
        super().__init__()

        init_t = float(torch.log(torch.tensor(1.0 / temperature_init))) #log(10)

        if learnable_temperature:
            self.t_prime = nn.Parameter(torch.tensor(init_t))
        else:
            self.register_buffer("t_prime", torch.tensor(init_t))

        if learnable_bias:
            self.bias = nn.Parameter(torch.tensor(bias_init))
        else:
            self.register_buffer("bias", torch.tensor(bias_init))

        assert mode is not None, 'Please, pass an explicit mode param.'
    def forward(self, img_emb: torch.Tensor, txt_emb: torch.Tensor) -> torch.Tensor:
        assert img_emb.dim() == 2 and txt_emb.dim() == 2, "Expected [B,D] embeddings"
        assert img_emb.shape == txt_emb.shape, f"Shape mismatch: {img_emb.shape} vs {txt_emb.shape}"

        # l2 normalization of embeddings
        img = F.normalize(img_emb, dim=-1)
        txt = F.normalize(txt_emb, dim=-1)

        # similarity matrix [B,B] -> diagonal entries = positive pairs, off-diagonal = negatives
        logits = img @ txt.t()

        # t = exp(t_prime)
        # logits = logits * t + bias
        if self.t_prime is not None:
            t = self.t_prime.exp().clamp(max=100) 
            logits = logits * t
        if self.bias is not None:
            logits = logits + self.bias

        # labels: 2 * eye(B) - ones(B)
        B = img.shape[0]


        match self.mode:
            case self.STANDARD:
                labels = 2 * torch.eye(B, device=logits.device, dtype=logits.dtype) - 1  
                loss = -F.logsigmoid(labels * logits).sum() / B
            case self.ONLY_POSITIVES:
                labels = 2 * torch.eye(B, device=logits.device, dtype=logits.dtype) - 1  
                loss = -F.logsigmoid(labels * logits)
                loss = (torch.eye(B, device=logits.device, dtype=logits.dtype) * loss).sum() / B 
            case self.CONTINUOUS:
                labels = txt @ txt.t()
                loss = -F.logsigmoid(labels * logits).sum() / B
            case _:
                raise RuntimeError(f'Invalid mode:{self.mode}')



        #version 2
        # loss = -F.logsigmoid(labels * logits)
        # mask = (report_idx[:, None] != report_idx[None, :]).float()
        # mask += torch.eye(B, device=logits.device, dtype=logits.dtype)  # keep positives
        # masked_loss = (mask * loss).sum() / mask.sum().clamp(min=1.0)



        return loss
