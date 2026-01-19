# Models

## 🧠 [ModelFactory.py](ModelFactory.py)
Creates model instances from config. The factory searches for model classes in the `models/` folder by looking up the dataset name in the module's namespace.

## Available Models

🧠 **[UNet2D.py](UNet2D.py):**
- 2D U-Net architecture
- Configurable: `in_channels`, `num_classes`, `init_features`

🧠 **[UNet3D.py](UNet3D.py):**
- 3D U-Net architecture  
- Configurable: `in_channels`, `num_classes`, `init_features`

## Adding Custom Models

1. Create a new file in `models/` with your model class that inherits from [`BaseModel`](../base/base_model.py) (or directly from `nn.Module`)
2. Implement `__init__()` to define layers
3. Implement `forward()` method that must at least return the output of the model
4. Add your model class to [`__init__.py`](__init__.py) in the `models/` folder so the ModelFactory can find it:
   ```python
   from .YourModel import YourModel
   ```
5. To use it, specify in config: `"model": {"type": "YourModel", "params": {...}}`
