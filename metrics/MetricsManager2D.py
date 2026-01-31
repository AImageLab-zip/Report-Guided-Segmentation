import os
import numpy as np
import pandas as pd
import torch
from monai.networks.utils import one_hot

class MetricsManager2D:
    """
    2D metrics manager that logs ONLY the lesion class (default index=1).
    Works with MONAI metrics such as DiceMetric and HausdorffDistanceMetric.

    Expected inputs:
      - prediction: logits [B, C, H, W] (float) OR indices [B, 1, H, W] (int)
      - label: indices [B, H, W] or [B, 1, H, W] (int, values in {0..C-1})

    It converts both prediction and label to binary two-channel one-hot:
      [B, 2, H, W] with channels = [background, lesion]
    then computes metrics and stores ONLY the lesion value.
    """

    def __init__(self, config, phase: str, lesion_index: int = 1, **metrics):
        assert phase in ["train", "val", "test"], f"phase must be train/val/test, got {phase}"
        self.config = config
        self.phase = phase
        self.metrics = metrics
        self.num_classes = int(self.config.model["num_classes"])
        self.lesion_index = int(lesion_index)

        # Resolve class names (optional)
        class_names = getattr(config, "classes", {}) if hasattr(config, "classes") else {}
        self.class_names = {int(k): v for k, v in class_names.items()} if class_names else {}
        self.class_names = {i: self.class_names.get(i, f"class_{i}") for i in range(self.num_classes)}
        self.lesion_name = self.class_names.get(self.lesion_index, "lesion")

        self.data = pd.DataFrame()
        self.flush()

    def flush(self):
        self.metric_sums = {}
        self.metric_counts = {}

    @staticmethod
    def _ensure_label_indices_2d(label: torch.Tensor) -> torch.Tensor:
        # Accept [B,H,W] or [B,1,H,W]
        if label.dim() == 3:
            return label.unsqueeze(1)
        if label.dim() == 4 and label.shape[1] == 1:
            return label
        raise ValueError(f"Label must be [B,H,W] or [B,1,H,W], got {tuple(label.shape)}")

    def _pred_to_indices_2d(self, prediction: torch.Tensor) -> torch.Tensor:
        # Accept indices [B,1,H,W] int
        if prediction.dtype in (torch.long, torch.int32, torch.int64) and prediction.dim() == 4 and prediction.shape[1] == 1:
            return prediction
        # Otherwise logits/probs [B,C,H,W]
        if prediction.dim() != 4:
            raise ValueError(f"Prediction must be [B,C,H,W] logits or [B,1,H,W] indices, got {tuple(prediction.shape)}")
        return torch.argmax(prediction, dim=1, keepdim=True)  # [B,1,H,W]

    def _to_binary_onehot(self, pred_idx: torch.Tensor, label_idx: torch.Tensor):
        """
        Convert multi-class indices to binary one-hot for lesion vs background.
        Output: pred_bin, label_bin of shape [B,2,H,W]
        """
        pred_oh = one_hot(pred_idx, num_classes=self.num_classes).float()    # [B,C,H,W]
        lab_oh  = one_hot(label_idx, num_classes=self.num_classes).float()   # [B,C,H,W]

        pred_les = pred_oh[:, self.lesion_index:self.lesion_index+1, ...]    # [B,1,H,W]
        lab_les  = lab_oh[:,  self.lesion_index:self.lesion_index+1, ...]    # [B,1,H,W]

        pred_bin = torch.cat([1 - pred_les, pred_les], dim=1)                # [B,2,H,W]
        lab_bin  = torch.cat([1 - lab_les,  lab_les],  dim=1)                # [B,2,H,W]
        return pred_bin, lab_bin

    def update_metrics(self, prediction: torch.Tensor, label: torch.Tensor):
        label_idx = self._ensure_label_indices_2d(label)
        pred_idx = self._pred_to_indices_2d(prediction)

        pred_bin, lab_bin = self._to_binary_onehot(pred_idx, label_idx)

        for metric_key, metric_func in self.metrics.items():
            value = metric_func(pred_bin, lab_bin)

            # MONAI often returns [B,2] or [2] or scalar depending on reduction.
            lesion_val = self._extract_lesion_value(value)

            # Store lesion-only
            class_key = f"{metric_key}_{self.lesion_name}"
            self.metric_sums[class_key] = self.metric_sums.get(class_key, 0.0) + lesion_val
            self.metric_counts[class_key] = self.metric_counts.get(class_key, 0) + 1

            # Store *_mean as alias for checkpoint logic (mean == lesion here)
            mean_key = f"{metric_key}_mean"
            self.metric_sums[mean_key] = self.metric_sums.get(mean_key, 0.0) + lesion_val
            self.metric_counts[mean_key] = self.metric_counts.get(mean_key, 0) + 1

    @staticmethod
    def _extract_lesion_value(value) -> float:
        """
        Given MONAI metric output, extract the lesion channel (channel index 1 in [bg,lesion]).
        """
        if not isinstance(value, torch.Tensor):
            return float(value)

        v = value.detach().cpu()
        if v.numel() == 0:
            return float("nan")

        # If [B,2] -> average over batch
        if v.dim() == 2 and v.shape[1] >= 2:
            v = torch.nanmean(v, dim=0)  # -> [2]
            return float(v[-1].item())

        # If [2]
        if v.dim() == 1 and v.shape[0] >= 2:
            return float(v[-1].item())

        # Scalar fallback
        return float(torch.nanmean(v).item())

    def compute_epoch_metrics(self, epoch: int):
        row = {"epoch": epoch}
        for k in self.metric_sums:
            row[k] = self.metric_sums[k] / max(1, self.metric_counts.get(k, 1))
        self.data = pd.concat([self.data, pd.DataFrame([row])], ignore_index=True)
        self.flush()

    def get_metric_at_epoch(self, metric_name: str, epoch: int):
        if metric_name not in self.data.columns and not any(self.data.columns.astype(str).str.startswith(metric_name)):
            raise ValueError(f"Metric {metric_name} not found.")

        epoch_data = self.data[self.data["epoch"] == epoch]
        if epoch_data.empty:
            raise ValueError(f"No data found for epoch {epoch}.")

        return epoch_data.filter(regex=f"^{metric_name}").to_dict(orient="records")[0]

    def save_to_csv(self, file_path: str):
        os.makedirs(file_path, exist_ok=True)
        self.data.to_csv(os.path.join(file_path, f"{self.phase}_metrics.csv"), index=False)

    def load_from_csv(self, file_path: str):
        load_path = os.path.join(file_path, f"{self.phase}_metrics.csv")
        if os.path.isfile(load_path):
            self.data = pd.read_csv(load_path)
            print(f"Loaded {self.phase} metrics from {load_path}")
        else:
            print(f"No {self.phase} metrics file found at {load_path}, starting fresh.")
