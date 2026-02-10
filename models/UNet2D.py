import torch.nn as nn
import torch.nn.functional as F
import torch
from base import BaseModel
from utils.pad_unpad import pad_to_2d, unpad_2d

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kwargs={'kernel_size': 3, 'stride': 1, 'padding': 1}):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **conv_kwargs)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.activ = nn.LeakyReLU()
        self.block = nn.Sequential(self.conv, self.norm, self.activ)

    def forward(self, x):
        return self.block(x)


# Single encoder block
class DownConvBlock(nn.Module):
    def __init__(self, in_channels : list, out_channels: list,
                 conv_kwargs={'kernel_size': 3, 'stride': 1, 'padding': 1}):
        super(DownConvBlock, self).__init__()

        assert len(in_channels) == len(out_channels), f"in_channels length is {len(in_channels)} while out_channels is {len(out_channels)}"

        # Variable number of convolutional block in each layer, based on the in_channels and out_channels length
        self.conv_blocks = nn.ModuleList(
            [ConvBlock(in_ch, out_ch, conv_kwargs) for in_ch, out_ch in zip(in_channels, out_channels)])

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        return self.pool(x), x


# Single decoder block
class UpConvBlock(nn.Module):
    def __init__(self, in_channels: list, out_channels: list, up_conv =True,
                 conv_kwargs={'kernel_size': 3, 'stride': 1, 'padding': 1},
                 upconv_kwargs={'kernel_size': 2, 'stride': 2}):
        super(UpConvBlock, self).__init__()

        assert len(in_channels) == len(out_channels), f"in_channels length is {len(in_channels)} while out_channels is {len(out_channels)}"

        # Variable number of convolutional block in each layer, based on the in_channels and out_channels length
        self.conv_blocks = nn.ModuleList(
            [ConvBlock(in_ch, out_ch, conv_kwargs) for in_ch, out_ch in zip(in_channels, out_channels)])

        self.up_conv = up_conv
        if self.up_conv:
            self.up_conv_op = nn.ConvTranspose2d(out_channels[-1], out_channels[-1], **upconv_kwargs)

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)

        if self.up_conv:
            return self.up_conv_op(x)
        else:
            return x


class UNet2D(BaseModel):
    def __init__(self, in_channels, num_classes, size=32, depth=3,t_prime_init = 0.1,bias_init = -10.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = num_classes
        self.size = size
        self.depth = depth

        self.encoder = nn.ModuleDict()
        self.encoder['0'] = DownConvBlock([self.in_channels, self.size], [self.size, self.size*2])
        for i in range(1, self.depth):
            self.encoder[str(i)] = DownConvBlock([self.size*(2**i), self.size*(2**i)],
                                              [self.size*(2**i), self.size*(2**(i+1))])

        self.bottleneck = UpConvBlock([self.size*(2**self.depth), self.size*(2**self.depth)],
                                        [self.size*(2**self.depth), self.size*(2**(self.depth+1))])

        self.decoder = nn.ModuleDict()
        for i in range(self.depth, 1, -1):
            self.decoder[str(i-1)] = UpConvBlock([self.size*(2**(i+1)) + self.size*(2**i), self.size*(2**i)],
                                            [self.size*(2**i), self.size*(2**i)])
        self.decoder['0'] = UpConvBlock([self.size*4 + self.size*2, self.size*2],
                                        [self.size*2, self.size*2],
                                        up_conv=False)
        self.out_layer = nn.Conv2d(self.size * 2, self.out_channels, kernel_size=1, stride=1, padding=0)

        self.bottleneck_channels = self.size * (2 ** (self.depth + 1))

        self.t_prime = nn.Parameter(torch.tensor(t_prime_init))
        self.bias_init = nn.Parameter(torch.tensor(bias_init))


    def forward(self, x, return_bottleneck: bool = False):
        #print(self.encoder)
        #print(self.bottleneck)
        #print(self.decoder)
        #print(self.out_layer)
        feat_list = []

        # Padding is needed if the size of the input is not divisible 'depth' times by 2
        pre_padding = (x.size(-1) % 2**self.depth != 0) or (x.size(-2) % 2**self.depth != 0) or (x.size(-3) % 2**self.depth != 0)
        if pre_padding:
            x, pads = pad_to_2d(x, 2**self.depth)
            #print(x.size())

        out, feat = self.encoder['0'](x)
        feat_list.append(feat)

        for block in list(self.encoder)[1:]:
            out, feat = self.encoder[block](out)
            feat_list.append(feat)

        bottleneck = self.bottleneck(out)
        out = bottleneck

        for block in self.decoder:
            out = self.decoder[block](torch.cat((out, feat_list[int(block)]), dim=1))
            del feat_list[int(block)]

        out = self.out_layer(out)

        if pre_padding:
            out = unpad_2d(out, pads)

        if return_bottleneck:
            return out, bottleneck
        return out

''' Example usage
model = UNet2D(3,4)
#x = torch.rand(1, 3, 248, 244, 64)
x = torch.rand(2, 3, 64, 64)
#x = torch.rand(1, 3, 32, 32, 32)
model.cuda()
x = x.to("cuda")
x = model.forward(x)
x = x.to("cpu")
print(x.size())
'''