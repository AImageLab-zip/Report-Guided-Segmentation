import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trainer import Trainer_3D, Trainer_2Dsliced
from trainer import Trainer_2D
import torch
from config import Config

def test_trainer3d():
    c = Config("/leonardo/home/userexternal/ocarpent/Report-Guided-Segmentation/config/config_brats3d.json")
    #print(c)

    #trainer = Trainer_3D(c, 1, True,"/work/grana_neuro/trained_models/ATLAS_2/3DUNet",resume=False, debug=True)
    trainer = Trainer_3D(config = c, 
                         epochs = 3, 
                         validation=True,
                         val_every=1,
                         save_path="/leonardo_work/IscrC_narc2/reports_project/trained_models/tests",
                         resume=False, 
                         debug=True, 
                         eval_metric_type='aggregated_mean', 
                         use_wandb=True)
    trainer.train()

def test_trainer2dsliced():
    c = Config("/work/grana_neuro/Brain-Segmentation/config/config_brats2d.json")
    #print(c)

    trainer = Trainer_2Dsliced(config = c, 
                               epochs = 3, 
                               val_every=1,
                               validation = True,
                               save_path = "/leonardo_work/IscrC_narc2/reports_project/trained_models/tests",
                               resume=False,
                               debug=True)
    trainer.train()

def test_trainer2d():
    c = Config("../config/config_qatacov2d.json")
    #print(c)

    trainer = Trainer_2D(config = c, 
                         epochs = 3, 
                         validation = True,
                         val_every = 1,
                         save_path="/leonardo_work/IscrC_narc2/reports_project/trained_models/tests",
                         resume=False,
                         debug=True)
    trainer.train()

#test_trainer3d()
#test_trainer2dsliced()
test_trainer3d()