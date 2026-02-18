import torch
import torch.nn as nn

from .UNet3D import UNet3D
from .TextGuidanceHead import ReportGuidanceHead
from utils.pad_unpad import pad_to_3d, unpad_3d


class UNet3DText(UNet3D):
    """
    UNet3D with optional report-guidance head.
    If `text_emb` is provided, forward returns (seg_logits, z_img, z_txt).
    Otherwise it returns only seg_logits.
    """

    def __init__(
        self,
        in_channels,
        num_classes,
        size=32,
        depth=3,
        text_dim=768,
        guidance_hidden_dim=1024,
        guidance_out_dim=512,
        text_proj=True,
        img_proj=True,
        t_prime_init=0.1,
        bias_init=-10.0,
    ):
        super().__init__(in_channels=in_channels, num_classes=num_classes, size=size, depth=depth)

        self.bottleneck_channels = self.size * (2 ** (self.depth + 1))
        self.guidance_head = ReportGuidanceHead(
            bottleneck_channels=self.bottleneck_channels,
            text_dim=text_dim,
            hidden_dim=guidance_hidden_dim,
            out_dim=guidance_out_dim,
            spatial_dims=3,
            text_proj=text_proj,
            img_proj=img_proj,
        )
        t_init = torch.log(torch.tensor(1.0 / t_prime_init))
        self.t_prime = nn.Parameter(t_init)
        self.bias = nn.Parameter(torch.tensor(bias_init, dtype=torch.float32))

    def forward(self, x, text_emb=None, return_emb: bool = False):
        # Handle both parameter names for backward compatibility
        return_features = return_emb 
        
        feat_list = []

        # Padding is needed if the size of the input is not divisible 'depth' times by 2
        pre_padding = (x.size(-1) % 2**self.depth != 0) or (x.size(-2) % 2**self.depth != 0) or (x.size(-3) % 2**self.depth != 0)
        if pre_padding:
            x, pads = pad_to_3d(x, 2**self.depth)
            #print(x.size())

        out, feat = self.encoder['0'](x)
        feat_list.append(feat)

        for block in list(self.encoder)[1:]:
            out, feat = self.encoder[block](out)
            feat_list.append(feat)

        bottleneck_feat = self.bottleneck(out)

        out = bottleneck_feat
        for block in self.decoder:
            out = self.decoder[block](torch.cat((out, feat_list[int(block)]), dim=1))
            del feat_list[int(block)]

        out = self.out_layer(out)

        # Ensure DDP tracks t_prime/bias in the forward graph
        # (these parameters are used in the loss outside forward)
        out = out + (self.t_prime * 0.0) + (self.bias * 0.0)

        if pre_padding:
            out = unpad_3d(out, pads)

        if text_emb is None and not return_features:
            return out
        elif text_emb is None and return_features:
            return out, bottleneck_feat
        elif text_emb is not None and not return_features:
            z_img, z_txt = self.guidance_head(bottleneck_feat, text_emb)
            return out, z_img, z_txt
        elif text_emb is not None and return_features:
            z_img, z_txt = self.guidance_head(bottleneck_feat, text_emb)
            return out, z_img, z_txt, bottleneck_feat