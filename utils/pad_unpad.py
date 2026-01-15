import torch
import torch.nn.functional as F
"""
This file contains functions to deal with odd dimensions in UNet
"""
def pad_to_3d(x, stride):
    d, h, w = x.shape[-3:]

    if d % stride > 0:
        new_d = d + stride - d % stride
    else:
        new_d = d
    if h % stride > 0:
        new_h = h + stride - h % stride
    else:
        new_h = h
    if w % stride > 0:
        new_w = w + stride - w % stride
    else:
        new_w = w
    ld, ud = int((new_d-d) / 2), int(new_d-d) - int((new_d-d) / 2)
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pads = (lw, uw, lh, uh, ld, ud)

    # zero-padding by default.
    # See others at https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.pad
    out = F.pad(x, pads, "constant", 0)

    return out, pads

def unpad_3d(x, pad):
    if pad[4] + pad[5] > 0:
        x = x[:, :, pad[4]:-pad[5], :, :]
    if pad[2] + pad[3] > 0:
        x = x[:, :, :, pad[2]:-pad[3], :]
    if pad[0] + pad[1] > 0:
        x = x[:, :, :, :, pad[0]:-pad[1]]
    return x


def pad_to_2d(x: torch.Tensor, stride: int):
    h, w = x.shape[-2:]

    new_h = h if h % stride == 0 else h + stride - (h % stride)
    new_w = w if w % stride == 0 else w + stride - (w % stride)

    top    = (new_h - h) // 2
    bottom = (new_h - h) - top
    left   = (new_w - w) // 2
    right  = (new_w - w) - left

    pads = (left, right, top, bottom)        
    x_pad = F.pad(x, pads, mode="constant", value=0)
    return x_pad, pads


def unpad_2d(x: torch.Tensor, pads):
    left, right, top, bottom = pads

    if top or bottom:
        end_h = -bottom if bottom > 0 else None
        x = x[:, :, top:end_h, :]

    if left or right:
        end_w = -right if right > 0 else None
        x = x[:, :, :, left:end_w]

    return x

inp = torch.randn(2, 3, 513, 505) 
padded, pads = pad_to_2d(inp, stride=32)
print(padded.shape)             

restored = unpad_2d(padded, pads)
assert torch.allclose(restored, inp)