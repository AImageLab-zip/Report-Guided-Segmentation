# Configuration

Contains JSON configuration files organized in two types:

**1. General Config (e.g., [config_atlas.json](config_atlas.json) and [config_brats2d.json](config_brats2d.json)):**
Defines model, dataset, training parameters, optimizer, loss, and metrics. This must be pass as a parameter to the main.py


```json
{
  "name": "experiment_name",
  "n_gpu": 1,

  "model": {"name": "UNet3D", "params": {...}},

  "dataset": {
    "name": "ATLAS",
    "root_folder": "/path/to/data",
    "transforms": "config/atlas_transforms.json", <-- path of Transforms Config
    ...
  },
  
  "classes":{"0": name_class_0, "1": name_class_1, ...},

  "loss": {"name": "DiceLoss", "loss_kwargs": {...}},
  "optimizer": {"name": "Adam", ...},
  "scheduler": {"name": "StepLR", ...},
  
  "metrics": [
    {
      "key": "DSC", <-- identification name of the metric
      "name": "DiceMetric", <-- name of the TorchIO/custom metric class
      "params": { "include_background": false }, <-- params for the metric constructor
      "train": true, <-- if metric must be computed during training phase
      "val": true, <-- if metric must be computed during validation phase 
      "test": true <-- if metric must be computed during test phase
    },
    ...
  ]
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

### Best model selection (`model_best.pth`)
The best checkpoint is selected using the **validation** value of the **first metric** in the `metrics` list (if *--validation* flag is passed to the [main.py](/main.py)).

- The metric being tracked is the `"<key>_mean"` value (mean across classes and validation data)
- It is assumed to be **maximized** (higher is better)

For example, if the first entry has `"key": "DSC"`, the trainer tracks `DSC_mean` on the validation set.
