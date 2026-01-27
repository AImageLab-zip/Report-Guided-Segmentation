import metrics
import monai

# Class to create a dictionary containing the metrics specified in the config file
class MetricsFactory:
    @staticmethod
    def create_instance(config, phase: str | None = None) -> dict:
        """
        Create instantiated metrics from config.

        Supports:
        - Old format: {"name": ["DiceMetric", ...]}
        - New format: [{"name": "DiceMetric", "params": {...}, "train": true/false, "val": ..., "test": ...}, ...]

        If `phase` is provided, only metrics with spec[phase] == True are returned.
        If the phase key is missing, default is False.
        """
        metrics_cfg = config.metrics

        # Backward-compatible parsing
        if isinstance(metrics_cfg, dict):
            metric_specs = [{"name": n, "params": {}} for n in metrics_cfg.get("name", [])]
        elif isinstance(metrics_cfg, list):
            metric_specs = metrics_cfg
        else:
            raise TypeError(f"`config.metrics` must be a dict (old format) or a list (new format), got: {type(metrics_cfg)}")

        if phase is not None:
            if phase not in ("train", "val", "test"):
                raise ValueError(f"`phase` must be one of ('train','val','test'), got: {phase}")

        metrics_dict = {}
        for spec in metric_specs:
            if not isinstance(spec, dict) or "name" not in spec:
                raise ValueError(f"Each metric spec must be a dict with at least a `name`. Got: {spec}")

            # Phase filtering (default missing => False)
            if phase is not None and not bool(spec.get(phase, False)):
                continue

            m_name = spec["name"]
            m_params = spec.get("params", {}) or {}
            if not isinstance(m_params, dict):
                raise ValueError(f"`params` for metric `{m_name}` must be a dict, got: {type(m_params)}")

            if m_name in monai.metrics.__dict__:
                m_class = getattr(monai.metrics, m_name)
                print(f"Loading {m_name} from Monai")
            elif m_name in metrics.__dict__:
                m_class = getattr(metrics, m_name)
            else:
                raise Exception(f"Could not find metric: {m_name}")

            try:
                metric = m_class(**m_params)
            except TypeError as e:
                raise TypeError(f"Could not instantiate {m_name} with params={m_params}\n{e}")

            key = spec.get("key")
            if not key:
                key = m_name if m_name not in metrics_dict else f"{m_name}_{len(metrics_dict)}"

            if key in metrics_dict:
                raise ValueError(f"Duplicate metric key `{key}`. Use unique `key` in config.")
            metrics_dict[key] = metric

        return metrics_dict