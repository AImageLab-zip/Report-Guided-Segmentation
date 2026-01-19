# Metrics

## 📊 `MetricsFactory.py`
Creates metric instances (supports [MONAI metrics](https://monai-dev.readthedocs.io/en/fixes-sphinx/metrics.html)).

## 📊 `MetricsManager.py`
Handles metric computation, accumulation, and storage.

**Functionality:**
- Computes metrics per iteration and accumulates them
- Handles both single-value metrics (losses) and multi-class metrics
- Automatically computes per-class and mean metrics
- Stores results in pandas DataFrame with one row per epoch

**Saved Format:**
CSV files (`train_metrics.csv`, `val_metrics.csv`, `test_metrics.csv`) with columns:
- `epoch`: Epoch number
- `<metric_name>`: For scalar metrics
- `<metric_name>_<class_name>`: For per-class metrics
- `<metric_name>_mean`: Mean across all classes

---

## Adding Custom Metrics

1. Create a new file in `metrics/` with your metric class
2. Add your metric class to [`__init__.py`](__init__.py) in the `metrics/` folder so the MetricsFactory can find it:
   ```python
   from .YourMetric import YourMetric
   ```
3. To use it, specify the class name in config: `"metrics": {"name": ["YourMetric"]}`
