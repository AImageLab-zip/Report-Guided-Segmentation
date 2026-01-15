import sys
import os
from trainer import Trainer_3D, Trainer_2Dsliced

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from config import Config

def test_trainer3d():
    c = Config("/work/grana_neuro/Brain-Segmentation/config/config_atlas.json")
    print(c)

    trainer = Trainer_3D(c, 1, True,"/work/grana_neuro/trained_models/ATLAS_2/3DUNet",resume=False, debug=True)
    trainer.train()

def test_trainer2dsliced():
    c = Config("/work/grana_neuro/Brain-Segmentation/config/config_brats2d.json")
    print(c)

    trainer = Trainer_2Dsliced(c, 1, True,"/work/grana_neuro/trained_models/BraTS23/2d",resume=False, debug=True)
    trainer.train()

test_trainer3d()
#test_trainer2dsliced()