# Optimizers

## 📈 `OptimizerFactory.py`
Creates PyTorch optimizers and learning rate schedulers from config. It looks up optimizer classes included in [torch.optim](https://docs.pytorch.org/docs/stable/optim.html), and scheduler in [torch.optim.lr_scheduler](https://docs.pytorch.org/docs/stable/optim.html) and instantiates them dynamically.
