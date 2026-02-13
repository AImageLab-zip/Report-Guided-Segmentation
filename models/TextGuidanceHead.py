import torch
import torch.nn as nn
import torch.nn.functional as F

class ReportGuidanceHead(nn.Module):
    def __init__(
        self,
        bottleneck_channels: int,
        text_dim: int = 768,
        hidden_dim: int = 1024, # to be set if both text_proj and img_proj are True
        out_dim: int = 512,  # to be set if both text_proj and img_proj are True
        spatial_dims: int = 2,
        text_proj: bool = True,
        img_proj: bool = True,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        if img_proj and not text_proj:
            self.out_dim = text_dim
        elif text_proj and not img_proj:
            self.out_dim = bottleneck_channels
        else:
            self.out_dim = out_dim

        self.img_proj = None
        self.txt_proj = None

        if self.spatial_dims == 2:
            self.img_pool = nn.AdaptiveAvgPool2d((1, 1))
            if img_proj:
                self.img_proj = nn.Conv2d(bottleneck_channels, self.out_dim, kernel_size=1)
        elif self.spatial_dims == 3:
            self.img_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            if img_proj:
                self.img_proj = nn.Conv3d(bottleneck_channels, self.out_dim, kernel_size=1)
        else:
            raise ValueError(f"Unsupported number of spatial dimensions: {spatial_dims}")

        if text_proj:
            self.txt_proj = nn.Sequential(
                nn.Linear(text_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, self.out_dim),
            )  # text: emb_dim -> hidden -> out
            

    def forward(self, feat, text_emb):
        if self.img_proj:
            feat = self.img_proj(feat)
        if feat.dim() == 4 and self.spatial_dims == 2:  # (B, C, H, W)
            z_img = self.img_pool(feat).squeeze(-1).squeeze(-1)  # (B, out_dim)
        elif feat.dim() == 5 and self.spatial_dims == 3:  # (B, C, D, H, W)
            z_img = self.img_pool(feat).squeeze(-1).squeeze(-1).squeeze(-1)  # (B, out_dim)
        else:
            raise ValueError(f"Unsupported number of dimensions or mismatch with spatial dimensions. "
                             f"Feature dims: {feat.dim()}, spatial dimensions value: {self.spatial_dims}")

        if self.txt_proj:
            z_txt = self.txt_proj(text_emb)
        else:
            z_txt = text_emb

        if z_img.shape[-1] != z_txt.shape[-1]:
            raise ValueError(
                f"Output dim mismatch: z_img has {z_img.shape[-1]}, z_txt has {z_txt.shape[-1]}. "
                "Enable projections or match input dims."
            )
        return z_img, z_txt
