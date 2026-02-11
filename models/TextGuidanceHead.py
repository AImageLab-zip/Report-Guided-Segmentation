import torch
import torch.nn as nn
import torch.nn.functional as F

class ReportGuidanceHead(nn.Module):

    TEXT_TO_IMAGE = 0            # Align text to image (default)
    IMAGE_TO_TEXT = 1            # Align image to text
    BOTH_TO_SHARED = 2           # Move both image and text embeddings to a shared space

    def __init__(
        self,
        bottleneck_channels: int,
        text_dim: int = 768,
        hidden_dim: int = 1024,
        mode: int = None,
    ):
        super().__init__()

        self.mode = mode

        # Always needed to collapse W, H dimension
        self.img_pool = nn.AdaptiveAvgPool2d((1, 1))

        match self.mode:
            case self.TEXT_TO_IMAGE:
                self.txt_proj = nn.Linear(text_dim, bottleneck_channels)
            case self.IMAGE_TO_TEXT:
                self.img_proj = nn.Conv2d(bottleneck_channels, text_dim, kernel_size=1)
            case self.BOTH_TO_SHARED:
                self.img_proj = nn.Conv2d(bottleneck_channels, hidden_dim, kernel_size=1)
                self.txt_proj = nn.Linear(text_dim, hidden_dim)
            case _:
                raise ValueError(f"Invalid mode for ReportGuidanceHead: {self.mode}")


    def forward(self, bottleneck_feat, text_emb):
        match self.mode:
            case self.TEXT_TO_IMAGE:
                z_img = self.img_pool(bottleneck_feat).squeeze(-1).squeeze(-1)  # [B, C]
                z_txt = self.txt_proj(text_emb)

            case self.IMAGE_TO_TEXT:
                img_vec = self.img_proj(bottleneck_feat)
                z_img = self.img_pool(img_vec).squeeze(-1).squeeze(-1)  # [B, C]
                z_txt = text_emb

            case self.BOTH_TO_SHARED:
                img_vec = self.img_proj(bottleneck_feat)
                z_img = self.img_pool(img_vec).squeeze(-1).squeeze(-1)  # [B, C]
                z_txt = self.txt_proj(text_emb)

        if z_img.shape != z_txt.shape:
            raise RuntimeError(
                f"Guidance embedding shape mismatch: image={tuple(z_img.shape)} text={tuple(z_txt.shape)}"
            )

        return z_img, z_txt
