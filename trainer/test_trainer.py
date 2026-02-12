import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trainer import Trainer_3D, Trainer_2Dsliced, Trainer_3DText
from trainer import Trainer_2D
import torch
from config import Config
import torch.distributed as dist

import os
import debugpy

def test_trainer3d():
    #c = Config("/work/grana_neuro/Brain-Segmentation/config/config_atlas.json")
    dist.init_process_group(backend="nccl", init_method="env://")
    if dist.get_rank() == 0:
        debugpy.listen(("0.0.0.0", 5679))
        debugpy.wait_for_client()
    
    dist.barrier()  # Ensure all processes wait for the debugger to attach before proceeding
    
    c = Config("/leonardo/home/userexternal/kmarches/Report-Guided-Segmentation/config/config_brats3d_multi.json")
    print(c)

    #trainer = Trainer_3D(c, 1, True,"/work/grana_neuro/trained_models/ATLAS_2/3DUNet",resume=False, debug=True)
    trainer = Trainer_3D(c, 3, True,1,"/leonardo_work/IscrC_narc2/reports_project/trained_models/BraTS3D/test",resume=False, debug=True, eval_metric_type='aggregated_mean', use_wandb=False, mixed_precision="fp16")
    trainer.train()

def test_trainer3d_text():
    #c = Config("/work/grana_neuro/Brain-Segmentation/config/config_atlas.json")
    dist.init_process_group(backend="nccl", init_method="env://")
    '''if dist.get_rank() == 0:
        debugpy.listen(("0.0.0.0", 5679))
        debugpy.wait_for_client()

    dist.barrier()'''  # Ensure all processes wait for the debugger to attach before proceeding
    c = Config("/leonardo/home/userexternal/kmarches/Report-Guided-Segmentation/config/config_brats3dtext_multi.json")
    print(c)

    #trainer = Trainer_3D(c, 1, True,"/work/grana_neuro/trained_models/ATLAS_2/3DUNet",resume=False, debug=True)
    trainer = Trainer_3DText(c, 3, True,1,"/leonardo_work/IscrC_narc2/reports_project/trained_models/BraTS3D/test",resume=False, debug=True, eval_metric_type='aggregated_mean', use_wandb=False, mixed_precision="fp16", pretrained_path="/leonardo_work/IscrC_narc2/reports_project/trained_models/BraTS3D/base_seg_depth4_s16_SGD_fp16/model_best.pth")

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
#test_trainer3d()
#test_trainer2d()
test_trainer3d_text()
