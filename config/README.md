# Configuration

Contains JSON configuration files organized in two types:

**1. General Config (e.g., [config_atlas.json](config_atlas.json) and [config_brats2d.json](config_brats2d.json)):**
Defines model, dataset, training parameters, optimizer, loss, and metrics. This must be pass as a parameter to the main.py


```json
{
  "name": "experiment_name",
  "model": {"type": "UNet3D", "params": {...}},
  "dataset": {
    "type": "ATLAS",
    "root_folder": "/path/to/data",
    "transforms": "config/atlas_transforms.json", <-- path of Transforms Config
    ...
  },
  "optimizer": {"type": "Adam", "args": {...}},
  "loss": {"name": "DiceLoss", "loss_kwargs": {...}},
  "metrics": {"name": ["DiceMetric"]}
}
```

**2. Transforms Config (e.g., [atlas_transforms.json](atlas_transforms.json) and [brats2d_transforms.json](brats2d_transforms.json)):**
Specifies preprocessing and augmentation pipelines using TorchIO transforms (to be passed in the general config under the key dataset.transforms).

```json
{
  "preprocessing": [...],
  "augmentations": [...]
}
```
