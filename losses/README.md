# Losses

## 🎯 `LossFactory.py`
Creates loss functions from config. Supports [MONAI losses](https://monai.readthedocs.io/en/1.4.0/losses.html) out-of-the-box.

## Adding Custom Losses

1. Create a new file in `losses/` with your loss class
2. Add your loss class to [`__init__.py`](__init__.py) in the `losses/` folder so the LossFactory can find it:
   ```python
   from .YourLoss import YourLoss
   ```
3. To use it, specify the class name in config: `"loss": {"name": "YourLoss", "loss_kwargs": {...}}`
