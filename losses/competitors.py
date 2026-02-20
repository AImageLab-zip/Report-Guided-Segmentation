import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class MFlagLoss(nn.Module):
    def __init__(self, lam_orth: float = 1.0):
        """
        Args:
            lam_orth: weight for orthogonality term
            normalize: whether to apply ℓ2 normalization to inputs
            reduction: "mean" or "sum" for alignment term
        """
        super().__init__()
        self.lam_orth = lam_orth

    def forward(self, img_emb_proj: torch.Tensor, txt_emb: torch.Tensor, img_emb_orig: torch.Tensor):
        """
        Args:
            img_emb_proj: (B, D)
            txt_emb: (B, D)
            img_emb_orig: (B, F)

        Returns:
            scalar loss
        """

        img_emb_proj = F.normalize(img_emb_proj, dim=1)
        txt_emb = F.normalize(txt_emb, dim=1)
        img_emb_orig = F.normalize(img_emb_orig, dim=1)

        # -------- Alignment loss --------
        align = 2 - 2 * (img_emb_proj * txt_emb).sum(dim=1)
        L_align = align.mean()


        # -------- Orthogonality loss --------
        # Gram matrix over feature dimension
        G = img_emb_orig.T @ img_emb_orig  # (D, D)

        D = G.size(0)
        I = torch.eye(D, device=G.device, dtype=G.dtype)

        # Equivalent to: sum((G - I)^2)
        L_orth = ((G - I) ** 2).sum()

        # -------- Total loss --------
        L_total = L_align + self.lam_orth * L_orth

        return L_total

def gather_features(features: torch.Tensor) -> torch.Tensor:
    """Gather features from all ranks. Gradients flow through local features only."""
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return features

    gathered = [torch.zeros_like(features) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, features)
    return torch.cat(gathered, dim=0)


class ConVIRTLoss(nn.Module):
    def __init__(self, temperature: float = 0.1, lam: float = 0.75):
        """
        Args:
            temperature: τ, softmax temperature.
            lam: λ, weighting between image-to-text and text-to-image loss.
                 λ=0.5 means equal weighting (symmetric), as in most practical use.
        """
        super().__init__()
        self.temperature = temperature
        self.lam = lam

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_features: (local_batch, dim)
            text_features:  (local_batch, dim)
        Returns:
            Scalar contrastive loss.
        """
        image_features = F.normalize(image_features, dim=-1)
        text_features  = F.normalize(text_features,  dim=-1)

        local_batch = image_features.shape[0]

        all_image = gather_features(image_features)
        all_text  = gather_features(text_features)

        logits_per_image = image_features @ all_text.T  / self.temperature  # l(v→u)
        logits_per_text  = text_features  @ all_image.T / self.temperature  # l(u→v)

        rank   = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        offset = rank * local_batch
        labels = torch.arange(offset, offset + local_batch, device=image_features.device)

        loss_v2u = F.cross_entropy(logits_per_image, labels)
        loss_u2v = F.cross_entropy(logits_per_text,  labels)

        return self.lam * loss_v2u + (1 - self.lam) * loss_u2v