import os
import torch
import torch.nn as nn

from base.base_trainer import BaseTrainer
from optimizers import OptimizerFactory
from losses import CombinedSegSigLoss


class BaseTrainerText(BaseTrainer):
    """
    Extends BaseTrainer with text-embedding support:
    - learnable projection from precomputed BioBERT embeddings to emb_dim
    - checkpointing of projection modules
    """

    def __init__(
        self,
        *args,
        use_text_guidance: bool = True,
        text_emb_dim_in: int = 768,
        emb_dim: int = 1024,
        text_emb_key: str = "text_emb",
        use_text_layernorm: bool = True,
        normalize_embeddings: bool = True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.use_text_guidance = use_text_guidance
        self.text_emb_dim_in = text_emb_dim_in
        config_emb_dim = None
        if hasattr(self.config, "model") and isinstance(self.config.model, dict):
            config_emb_dim = self.config.model.get("emb_dim")
        self.emb_dim = config_emb_dim if config_emb_dim is not None else emb_dim
        self.text_emb_key = text_emb_key
        self.normalize_embeddings = normalize_embeddings
        self._skip_text_norm = isinstance(self.loss, CombinedSegSigLoss)
        
        self.loss = self.loss.to(self.device)

        # --- Learnable projection head for text: 768 -> emb_dim (1024) ---
        if self.use_text_guidance:
            self.text_proj = nn.Linear(self.text_emb_dim_in, self.emb_dim).to(self.device)
            self.text_ln = nn.LayerNorm(self.emb_dim).to(self.device) if use_text_layernorm else None

            # Rebuild optimizer + scheduler so that they include text params
            self._rebuild_optimizer_with_text()
            
        else:
            self.text_proj = None
            self.text_ln = None
            
    def _rebuild_optimizer_with_text(self):
        """
        Recreate optimizer/scheduler to include model + text head parameters.
        Uses same config (no config changes needed).
        """
        extra_params = list(self.text_proj.parameters())
        if self.text_ln is not None:
            extra_params += list(self.text_ln.parameters())

        # Create a fresh optimizer + scheduler with extra params
        self.optimizer, self.lr_scheduler = OptimizerFactory.create_instance(
            self.model, self.config, extra_params=extra_params
        )

    def project_text(self, text_emb: torch.Tensor) -> torch.Tensor:
        """
        text_emb: [B, text_emb_dim_in]
        returns:  [B, emb_dim]
        """
        if not self.use_text_guidance:
            raise RuntimeError("use_text_guidance=False but project_text was called.")

        z = self.text_proj(text_emb)
        if self.text_ln is not None:
            z = self.text_ln(z)

        if self.normalize_embeddings and not self._skip_text_norm:
            z = torch.nn.functional.normalize(z, dim=-1)

        return z

    # -------------------- Checkpointing --------------------

    def _save_checkpoint(self, epoch, save_best):
        """
        Extend BaseTrainer checkpoint to include text projection weights.
        """
        state = {
            'name': type(self.model).__name__,
            'config': self.config,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            'epoch': epoch,
            'best_metric': self.best_metric,
            'wandb_run_id': self.wandb_run_id if self.use_wandb else None,

            'use_text_guidance': self.use_text_guidance,
            'emb_dim': self.emb_dim,
            'text_emb_dim_in': self.text_emb_dim_in,
            'text_emb_key': self.text_emb_key,
            'normalize_embeddings': self.normalize_embeddings,
            'text_proj': self.text_proj.state_dict() if self.text_proj is not None else None,
            'text_ln': self.text_ln.state_dict() if self.text_ln is not None else None,
        }

        checkpoint_path = os.path.join(self.save_path, 'model_last.pth')
        torch.save(state, checkpoint_path)

        if save_best:
            checkpoint_path = os.path.join(self.save_path, 'model_best.pth')
            print(f'Saving checkpoints {checkpoint_path}')
            torch.save(state, checkpoint_path)

    def _resume_checkpoint(self):
        """
        Extend BaseTrainer resume to restore text projection weights (if present).
        """
        """
        Custom resume that assumes optimizer already includes text params.
        """
        checkpoint_path = os.path.join(self.save_path, 'model_last.pth')
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model weights
        self.model.load_state_dict(checkpoint['model'])
        print("Model weights loaded.")

        # Load text head weights if present
        if checkpoint.get("use_text_guidance", False) and self.use_text_guidance:
            if checkpoint.get("text_proj") is not None and self.text_proj is not None:
                self.text_proj.load_state_dict(checkpoint["text_proj"])
                print("Text projection weights loaded.")
            if checkpoint.get("text_ln") is not None and self.text_ln is not None:
                self.text_ln.load_state_dict(checkpoint["text_ln"])
                print("Text LayerNorm weights loaded.")

        # Load optimizer state (now compatible because optimizer was built with same params)
        if 'optimizer' in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Optimizer state loaded.")

        # Load scheduler state
        if 'lr_scheduler' in checkpoint and self.lr_scheduler is not None and checkpoint['lr_scheduler'] is not None:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            print("Learning rate scheduler state loaded.")

        # Epoch/best metric
        self.start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {self.start_epoch}.")
        self.best_metric = checkpoint.get('best_metric', 0)
        print(f"Best metric so far: {self.best_metric}")

        # Metrics
        self.train_metrics.load_from_csv(self.save_path)
        self.test_metrics.load_from_csv(self.save_path)
        if self.validation:
            self.val_metrics.load_from_csv(self.save_path)

        # W&B resume
        if self.use_wandb:
            self.wandb_run_id = checkpoint.get('wandb_run_id', None)
            if self.wandb_run_id:
                self._init_wandb(resume_id=self.wandb_run_id)
            else:
                print("Warning: No W&B run ID found in checkpoint. Starting new run.")
                self._init_wandb()
