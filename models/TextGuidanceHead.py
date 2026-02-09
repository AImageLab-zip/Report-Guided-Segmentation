import torch
import torch.nn as nn
import torch.nn.functional as F

class ReportGuidanceHead(nn.Module):
    def __init__(self, bottleneck_channels: int, text_dim: int = 768, hidden_dim: int = 1024, out_dim: int = 512, spatial_dims: int = 2):
        super().__init__()
        self.spatial_dims = spatial_dims
        if self.spatial_dims == 2:
            self.img_proj = nn.Conv2d(bottleneck_channels, out_dim, kernel_size=1)
            self.img_pool = nn.AdaptiveAvgPool2d((1, 1))        
        elif self.spatial_dims == 3: 
            self.img_proj = nn.Conv3d(bottleneck_channels, out_dim, kernel_size=1)
            self.img_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            raise ValueError(f"Unsupported number of spatial dimensions: {spatial_dims}")
        
        self.txt_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        ) # text: emb_dim -> hidden -> out

    def forward(self, feat, text_emb):
        feat = self.img_proj(feat)
        if feat.dim() == 4: 
            z_img = self.img_pool(feat).squeeze(-1).squeeze(-1)  # (B, out_dim)
        elif feat.dim() == 5: 
            z_img = self.img_pool(feat).squeeze(-1).squeeze(-1).squeeze(-1)  # (B, out_dim)
        else:
            raise ValueError(f"Unsupported number of dimensions: {feat.dim()}")

        z_txt = self.txt_proj(text_emb)
        return z_img, z_txt
