import os
import torch
import torch.nn as nn

from base.base_trainer import BaseTrainer
from optimizers import OptimizerFactory

from models.TextGuidanceHead import ReportGuidanceHead


class BaseTrainerText(BaseTrainer):
    """
    Extends BaseTrainer with report-guided training support:
    - uses precomputed BioBERT embeddings (e.g., 768-d)
    - adds a learnable ReportGuidanceHead that:
        (1) pools U-Net bottleneck -> image vector
        (2) projects image + text into a shared embedding space
    - checkpointing/resume includes guidance head
    """

    def __init__(
        self,
        *args,
        use_text_guidance: bool = True,
        text_emb_dim_in: int = None,
        guidance_hidden_dim: int = 1024,   # internal MLP hidden
        guidance_mode: int = None,
        text_emb_key: str = "text_emb",
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        # allow finetuning from pretrained weights if provided
        pretrained_path = getattr(self.config, "pretrained_unet_path", None)
        if pretrained_path:
            if not os.path.exists(pretrained_path):
                raise FileNotFoundError(f"pretrained_unet_path not found: {pretrained_path}")
            ckpt = torch.load(pretrained_path, map_location="cpu")

            # Support both "raw state_dict" and "trainer checkpoint dict"
            state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            print(f"[Pretrain] Loaded U-Net from {pretrained_path}")
            print(f"[Pretrain] Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
        else:
            print("[Pretrain] No pretrained_unet_path provided. Training from scratch.")


        self.use_text_guidance = use_text_guidance
        self.text_emb_dim_in = self.config.trainer.get("text_emb_dim_in", text_emb_dim_in)
        self.text_emb_key = text_emb_key

        if hasattr(self.config, "trainer"):
            guidance_hidden_dim = self.config.trainer.get("guidance_hidden_dim", guidance_hidden_dim)
            guidance_mode = self.config.trainer.get("guidance_mode", guidance_mode)

        self.guidance_hidden_dim = guidance_hidden_dim
        self.guidance_mode = guidance_mode

        self.loss = self.loss.to(self.device)

        if self.use_text_guidance:
            if hasattr(self.model, "bottleneck_channels"):
                bottleneck_channels = int(self.model.bottleneck_channels)
            else:
                raise AttributeError(
                    "Your model must expose bottleneck channel dimension as `model.bottleneck_channels` "
                    "(or modify BaseTrainerText to read the correct attribute)."
                )

            self.guidance_head = ReportGuidanceHead(
                bottleneck_channels=bottleneck_channels,
                text_dim=self.text_emb_dim_in,
                hidden_dim=self.guidance_hidden_dim,
                mode=self.guidance_mode,
            ).to(self.device)

            # Rebuild optimizer + scheduler so that they include guidance head params
            self._rebuild_optimizer_with_guidance()

        else:
            self.guidance_head = None

    # -------------------- Optimizer --------------------

    def _rebuild_optimizer_with_guidance(self):
        """
        Recreate optimizer/scheduler to include model + guidance head parameters.
        Uses the same config (no config changes needed).
        """
        extra_params = list(self.guidance_head.parameters())
        self.optimizer, self.lr_scheduler = OptimizerFactory.create_instance(
            self.model, self.config, extra_params=extra_params
        )

    # -------------------- Checkpointing --------------------

    def _save_checkpoint(self, epoch, save_best):
        """
        Extend BaseTrainer checkpoint to include guidance head weights.
        """
        state = {
            'name': type(self.model).__name__,
            'config': self.config,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict() if self.optimizer is not None else None,
            'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            'epoch': epoch,
            'best_metric': self.best_metric,
            'wandb_run_id': self.wandb_run_id if self.use_wandb else None,

            'use_text_guidance': self.use_text_guidance,
            'text_emb_dim_in': self.text_emb_dim_in,
            'text_emb_key': self.text_emb_key,
            'guidance_hidden_dim': self.guidance_hidden_dim,
            'guidance_mode': self.guidance_mode,

            'guidance_head': self.guidance_head.state_dict() if self.guidance_head is not None else None,
        }

        checkpoint_path = os.path.join(self.save_path, 'model_last.pth')
        torch.save(state, checkpoint_path)

        if save_best:
            checkpoint_path = os.path.join(self.save_path, 'model_best.pth')
            print(f'Saving checkpoints {checkpoint_path}')
            torch.save(state, checkpoint_path)

    def _resume_checkpoint(self):
        """
        Extend BaseTrainer resume to restore guidance head weights (if present).
        Assumes optimizer already includes guidance params (we rebuild in __init__).
        """
        checkpoint_path = os.path.join(self.save_path, 'model_last.pth')
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model weights
        self.model.load_state_dict(checkpoint['model'])
        print("Model weights loaded.")

        # Load guidance head weights if present
        if checkpoint.get("use_text_guidance", False) and self.use_text_guidance:
            if checkpoint.get("guidance_head") is not None and self.guidance_head is not None:
                self.guidance_head.load_state_dict(checkpoint["guidance_head"])
                print("Guidance head weights loaded.")

        # Load optimizer state
        if checkpoint.get('optimizer') is not None and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Optimizer state loaded.")

        # Load scheduler state
        if checkpoint.get('lr_scheduler') is not None and self.lr_scheduler is not None:
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
