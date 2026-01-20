import sys
import os

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import datasets
from ModelFactory import ModelFactory #add . if yu use pytest
from config import Config
import torchio as tio
from tqdm import tqdm

def test_unet3d():
    m = ModelFactory()
    c = Config("/work/grana_neuro/Brain-Segmentation/config_atlas.json") # Place here the path to your config file

    unet3d = m.create_instance(c).to("cuda")
    # Input tensor must be in the format (batch_size, n_channels, H, W, D)
    x = torch.rand(4, 3, 64, 64, 40)
    y = torch.rand(2, 3, 58, 58, 31)
    x = x.to("cuda")
    y = y.to("cuda")

    out = unet3d.forward(x)
    print(x.size(), out.size())

    assert out.size() == (x.size(0), c.model['num_classes'], x.size(2), x.size(3), x.size(4)), \
        f"input size mismatch with output size, input is {x.size()} while output {out.size()}"

    out = unet3d.forward(y)
    print(y.size(), out.size())

    assert out.size() == (y.size(0), c.model['num_classes'], y.size(2), y.size(3), y.size(4)), \
        f"input size mismatch with output size, input is {y.size()} while output {out.size()}"


if __name__ == '__main__':
    test_unet3d()