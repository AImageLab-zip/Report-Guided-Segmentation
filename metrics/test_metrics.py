import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from monai.metrics import DiceMetric
import torch
from MetricsManager import MetricsManager #togli il . se non usi pytest
from monai.utils import MetricReduction
from monai.losses import DiceLoss
import torch.nn.functional as F
from config import Config

def test_metrics():
    # Example predictions and labels
    prediction = torch.randn((2, 3, 32, 32, 16))  # Batch of 2, 3 classes
    label = torch.randint(0, 3, (2, 1, 32, 32, 16))  # Ground truth
    c = Config("/work/grana_neuro/Brain-Segmentation/config/config_atlas.json")
    # Remove the channel dimension if necessary (squeeze the 1)
    label_squeezed = label.squeeze(1)  # Shape: [batch_size, depth, height, width]

    # Apply one-hot encoding
    label_one_hot = F.one_hot(label_squeezed,
                              num_classes=3)  # Shape: [batch_size, depth, height, width, num_classes]

    # Permute the dimensions to match the format [batch_size, num_classes, depth, height, width]
    label_one_hot = label_one_hot.permute(0, 4, 1, 2, 3).float()  # Convert to float if necessary

    # Initialize th
    #
    #
    # e metrics manager with DiceMetric
    metrics_manager = MetricsManager(config=c, phase='train', DiceMetric=DiceMetric(include_background=True, reduction="none", num_classes=3), Loss=DiceLoss())

    for i in range (1, 3):
        metrics_manager.update_metrics(prediction, label_one_hot)

    # Log to W&B
    #metrics_manager.log_to_wandb(step=1)

    # Get the metrics for a specific epoch
    metrics_manager.compute_epoch_metrics(epoch=1)
    metrics = metrics_manager.get_metric_at_epoch('DiceMetric', epoch=1)
    print(metrics)

    # Save and load the metrics to/from CSV
    metrics_manager.save_to_csv('/work/grana_neuro/trained_models/ATLAS_2/3DUNet/')
    metrics_manager.load_from_csv('metrics.csv')

    # Print the DataFrame
    print(metrics_manager.data)

test_metrics()