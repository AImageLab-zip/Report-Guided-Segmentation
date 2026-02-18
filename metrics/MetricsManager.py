import pandas as pd
import wandb
import numpy as np
import torch
import os
from monai.networks.utils import one_hot
import torch.distributed as dist
class MetricsManager:
    """
    Class to manage metrics/losses computation, storage, and logging on wandb.
    """

    def __init__(self, config, phase, **metrics):
        """
        Initialize the MetricsManager with metrics/losses, the phase (train/val/test), and config.
        :param config: config file
        :param phase: train/val/test
        """
        assert phase in ['train', 'val', 'test'], f'phase should be train, val, or test, passed: {phase}'
        self.metrics = metrics # Dictionary containing the metric classes
        self.phase = phase
        self.config = config

        self.num_classes = self.config.model['num_classes']
        if hasattr(config, 'classes'):
            self.class_names = config.classes
        else:
            self.class_names = {}
        self.class_names = {int(k): v for k, v in self.class_names.items()}

        # Fill class_names with default names for missing classes
        if len(self.class_names) != self.num_classes:
            self.class_names = {i: self.class_names.get(i, f'class_{i}') for i in range(self.num_classes)}

        # Load aggregated regions configuration (for BraTS: ET, TC, WT)
        self.aggregated_regions = {}
        if hasattr(config, 'aggregated_regions'):
            self.aggregated_regions = config.aggregated_regions

        # Initialize DataFrame, where columns are for each metric/class and rows for each epoch
        self.data = pd.DataFrame()
        # Initialize accumulators for metrics
        self.metric_sums = {}
        self.metric_counts = {}
        self.flush()

    def flush(self):
        """
        Resets the metric accumulators to zero.
        """
        self.metric_sums = {}
        self.metric_counts = {}

    def update_metrics(self, prediction, label):
        """
        Update the accumulators with metrics from a single iteration.
        Compute all the metrics/losses for the given prediction and label.
        Handles both single-value metrics (e.g., loss) and multiclass metrics.
        Also appends the mean value of the metric if it is multiclass.
        """
        for metric_name, metric_func in self.metrics.items():
            if metric_name.lower() == 'combinedsegsigloss':
                continue
            if "loss" not in metric_name.lower():
                # For metrics: convert to one-hot for comparison
                if prediction.dtype in (torch.long, torch.int, torch.int32, torch.int64):
                    # If the pred already "passed" through argmax just convert to one-hot
                    pred = one_hot(prediction, num_classes=self.num_classes).float()
                else:
                    # Raw logits - apply softmax and argmax
                    pred = torch.softmax(prediction, dim=1)  # [B, C, H, W, D]
                    pred = torch.argmax(pred, dim=1, keepdim=True)  # [B, 1, H, W, D]
                    pred = one_hot(pred, num_classes=prediction.shape[1]).float()  # [B, C, H, W, D]
                # For metrics, ensure label is one-hot
                label_proc = label if label.dtype != torch.long and label.shape[1] > 1 else one_hot(label, num_classes=self.num_classes).float()
            else:
                # For loss functions: keep raw predictions (logits) but convert labels to one-hot
                pred = prediction
                # Convert label to one-hot if it's not already
                label_proc = label if label.dtype != torch.long and label.shape[1] > 1 else one_hot(label, num_classes=self.num_classes).float()
            
            # Use the processed prediction and label to compute the metric/loss
            value = metric_func(pred, label_proc)

            # For multiclass metrics (tensors with dimension >= 1)
            if isinstance(value, torch.Tensor):
                if value.dim() >= 1:
                    # Convert to numpy array
                    value_np = value.cpu().numpy()
                    # If necessary, expand singleton dimensions
                    if value_np.shape[0] == 1 and self.num_classes is not None:
                        value_np = np.repeat(value_np, self.num_classes, axis=0)
                    value_np = np.nanmean(value_np, axis=0)
                    idx_offset = 1 if len(value_np) < self.num_classes else 0
                    for idx, val in enumerate(value_np, start=idx_offset):
                        class_name = self.class_names.get(idx, f'class_{idx}')
                        key = f'{metric_name}_{class_name}'
                        self.metric_sums[key] = self.metric_sums.get(key, 0.0) + val
                        self.metric_counts[key] = self.metric_counts.get(key, 0) + 1
                    # Update mean metric across all classes
                    mean_value = np.nanmean(value_np)
                    key = f'{metric_name}_mean'
                    self.metric_sums[key] = self.metric_sums.get(key, 0.0) + mean_value
                    self.metric_counts[key] = self.metric_counts.get(key, 0) + 1
                else:
                    # Single value tensor
                    val = value.item()
                    self.metric_sums[metric_name] = self.metric_sums.get(metric_name, 0.0) + val
                    self.metric_counts[metric_name] = self.metric_counts.get(metric_name, 0) + 1
            else:
                # Non-tensor metric
                self.metric_sums[metric_name] = self.metric_sums.get(metric_name, 0.0) + value
                self.metric_counts[metric_name] = self.metric_counts.get(metric_name, 0) + 1
        
        # Compute aggregated region metrics if configured
        if self.aggregated_regions:
            self.compute_aggregated_metrics(prediction, label)

    def compute_aggregated_metrics(self, prediction, label):
        """
        Compute metrics on aggregated regions (e.g., BraTS: ET, TC, WT).
        Aggregated regions are defined as combinations of classes.
        """
        for metric_name, metric_func in self.metrics.items():
            # Skip loss functions for aggregated metrics
            if "loss" in metric_name.lower():
                continue
            
            region_values = []
            for region_name, class_indices in self.aggregated_regions.items():
                # Convert prediction and label to one-hot if needed
                if prediction.dtype in (torch.long, torch.int, torch.int32, torch.int64):
                    pred = one_hot(prediction, num_classes=self.num_classes).float()
                else:
                    pred = torch.softmax(prediction, dim=1)
                    pred = torch.argmax(pred, dim=1, keepdim=True)
                    pred = one_hot(pred, num_classes=prediction.shape[1]).float()
                
                label_proc = label if label.dtype != torch.long and label.shape[1] > 1 else one_hot(label, num_classes=self.num_classes).float()
                
                # Aggregate the specified classes by summing their channels
                # Result: binary mask where 1 = any of the specified classes present
                pred_region = torch.sum(pred[:, class_indices, ...], dim=1, keepdim=True).clamp(0, 1)
                label_region = torch.sum(label_proc[:, class_indices, ...], dim=1, keepdim=True).clamp(0, 1)
                
                # Convert to binary one-hot format [B, 2, ...] for background and region
                pred_binary = torch.cat([1 - pred_region, pred_region], dim=1)
                label_binary = torch.cat([1 - label_region, label_region], dim=1)
                
                # Compute metric on aggregated region
                value = metric_func(pred_binary, label_binary)
                
                if isinstance(value, torch.Tensor):
                    if value.dim() >= 1:
                        # Take only the foreground (region) metric, not background
                        val = value.cpu().numpy()[-1]  # Last element is the region
                        if isinstance(val, np.ndarray):
                            val = float(val.item()) if val.size == 1 else float(val)
                        else:
                            val = float(val)
                    else:
                        val = value.item()
                else:
                    val = float(value) if not isinstance(value, (int, float)) else value

                # Skip NaNs (e.g., empty region with ignore_empty=True)
                if isinstance(val, float) and np.isnan(val):
                    continue
                
                key = f'{metric_name}_{region_name}'
                self.metric_sums[key] = self.metric_sums.get(key, 0.0) + val
                self.metric_counts[key] = self.metric_counts.get(key, 0) + 1
                region_values.append(val)
            
            # Compute mean across all aggregated regions
            if region_values:
                mean_value = np.nanmean(region_values)
                key = f'{metric_name}_aggregated_mean'
                self.metric_sums[key] = self.metric_sums.get(key, 0.0) + mean_value
                self.metric_counts[key] = self.metric_counts.get(key, 0) + 1

    def compute_epoch_metrics(self, epoch):
        """
        Compute globally-correct mean metrics for the epoch.
        Identical results are stored on every rank.
        """
        row_data = {'epoch': epoch}
        if dist.is_available() and dist.is_initialized():
            torch.cuda.set_device(dist.get_rank())  # Ensure each process uses the correct GPU

            keys = list(self.metric_sums.keys())
            world_size = dist.get_world_size()
            if world_size > 1:
                all_keys = [None for _ in range(world_size)]
                dist.all_gather_object(all_keys, keys)
                keys = sorted({k for sub in all_keys for k in sub})
        else:
            keys = list(self.metric_sums.keys())

        for key in keys:

            # local --> tensor
            s = torch.tensor(self.metric_sums.get(key, 0.0), device="cuda", dtype=torch.float64)
            c = torch.tensor(self.metric_counts.get(key, 0.0), device="cuda", dtype=torch.float64)

            # global reduction
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(s, op=dist.ReduceOp.SUM)
                dist.all_reduce(c, op=dist.ReduceOp.SUM)

            # overwrite local state with global values
            self.metric_sums[key] = s.item()
            self.metric_counts[key] = max(1.0, c.item())

            # compute global mean
            mean_value = self.metric_sums[key] / self.metric_counts[key]
            row_data[key] = mean_value

        # identical row appended on all ranks
        self.data = pd.concat(
            [self.data, pd.DataFrame([row_data])],
            ignore_index=True
        )

        # reset for next epoch
        self.flush()

    def get_metric_at_epoch(self, metric_name, epoch):
        """
        Retrieve the metric values at a specific epoch.
        """
        if metric_name not in self.data.columns and not any(self.data.columns.astype(str).str.startswith(metric_name)):
            raise ValueError(f'Metric {metric_name} not found in the stored data.')

        # Filter the row for the given epoch
        epoch_data = self.data[self.data['epoch'] == epoch]
        if epoch_data.empty:
            raise ValueError(f'No data found for epoch {epoch}.')

        # Return the data for the requested metric and epoch
        result = epoch_data.filter(regex=f'^{metric_name}').to_dict(orient='records')[0]
        return result

    def log_to_wandb(self, epoch: int = None):
        """
        Log the metrics to Weights & Biases for the current phase.
        """
        if dist.get_rank() == 0:
            if not self.data.empty:
                # Get the most recent row (current epoch)
                row = self.data.iloc[-1]
                log_dict = {}
                if epoch:
                    log_dict[f"{self.phase}/epoch"] = epoch
                for col in self.data.columns:
                    log_dict[f"{self.phase}/{col}"] = row[col]
                
                wandb.log(log_dict)
            else:
                print("No data to log.")

    def save_to_csv(self, file_path):
        """
        Save the metrics DataFrame to a CSV file.
        """
        if dist.get_rank()==0:
            self.data.to_csv(os.path.join(file_path, f'{self.phase}_metrics.csv'), index=False)

    def load_from_csv(self, file_path):
        """
        Load the metrics from a CSV file.
        """
        load_path = os.path.join(file_path, f'{self.phase}_metrics.csv')
        if os.path.isfile(load_path):
            self.data = pd.read_csv(load_path)
            print(f"Loaded validation metrics from {load_path}")
        else:
            print(f"No validation metrics file found at {load_path}, starting fresh.")

    def compute_averages(self):
        """
        Compute and return a dictionary of average values for all metrics.
        """
        averages = self.data.mean(numeric_only=True)
        return averages.to_dict()
