# Trainers

## Available Trainers

🏋🏼‍♂️ **[trainer_3D.py](trainer_3D.py):**

Trainer for 3D volume segmentation.

**Key Methods:**
- `_train_epoch(epoch)`: Iterates through patch-based training loader, computes loss, backpropagates, updates metrics
- `eval_epoch(epoch, phase)`: Performs patch-based inference on full volumes using GridSampler/GridAggregator, computes validation/test metrics
- `_inference_sampler(sample)`: Creates GridSampler for dense patch-based inference
- `_results_dict(phase, epoch)`: Retrieves metrics for the epoch

🏋🏼‍♂️ **[trainer_2Dsliced.py](trainer_2Dsliced.py):**

Trainer for 2D slice-based segmentation.

## Implementing a Custom Trainer

1. Create a new file in `trainer/` with your trainer class that inherits from [`BaseTrainer`](../base/base_trainer.py)
2. Implement `_train_epoch(epoch)`: Define training loop, iterate through batches, compute loss, backpropagate, and return metrics dict
3. Implement `eval_epoch(epoch, phase)`: Define validation/test loop, compute predictions and metrics, and return metrics dict
4. Add your trainer class to [`__init__.py`](__init__.py) in the `trainer/` folder:
   ```python
   from .YourTrainer import YourTrainer
   ```
5. Use it by specifying `--trainer YourTrainer` when running `main.py`, or let it auto-detect based on your dataset type
