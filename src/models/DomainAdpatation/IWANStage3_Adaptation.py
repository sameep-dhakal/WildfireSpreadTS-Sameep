import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss
from itertools import cycle
import torchmetrics

from models.SMPModel import SMPModel
from models.DomainAdpatation.IWANStage2_WeightEstimator import (
    DomainHead3x1024,
    DomainheadCNN,
)
from ..BaseModel import BaseModel

class IWANStage3_Adaptation(BaseModel):
    def __init__(
        self,
        encoder_name: str,
        n_channels: int,
        pos_class_weight: float,
        save_dir: str = "/develop/results/", # Add this line back in
        stage1_ckpt: str = None,
        stage2_ckpt: str = None,
        gamma_entropy: float = 0.1,
        lr: float = 1e-4,
        *args,
        **kwargs,
    ):
        super().__init__(
            n_channels=n_channels,
            flatten_temporal_dimension=True,
            pos_class_weight=pos_class_weight,
            *args,
            **kwargs,
        )
        self.save_hyperparameters()
        self.save_hyperparameters(ignore=["stage1_ckpt", "stage2_ckpt"])
        self.save_dir = save_dir # Now the CLI can "see" the destination
        
        # 1. Load Architecture & Weights
        base_seg = SMPModel.load_from_checkpoint(stage1_ckpt, encoder_name=encoder_name)
        self.encoder = base_seg.model.encoder
        self.decoder = base_seg.model.decoder
        self.seg_head = base_seg.model.segmentation_head

        # Store initial encoder weights to see how much we "drift" away from Stage 1
        self.register_buffer("initial_encoder_flat", 
                             torch.cat([p.flatten() for p in self.encoder.parameters()]).detach())

        # 2. Load Importance Oracle (Stage 2)
        state2 = torch.load(stage2_ckpt, map_location="cpu")
        feat_dim = state2.get("feat_dim", self.encoder.out_channels[-1])
        try:
            self.domain_head = DomainHead3x1024(feat_dim)
            self.domain_head.load_state_dict(state2["discriminator"])
        except:
            self.domain_head = DomainheadCNN(in_channels=feat_dim)
            self.domain_head.load_state_dict(state2["discriminator"])

        for p in self.domain_head.parameters():
            p.requires_grad = False

        # 3. Validation Metrics (Specifically for Target Domain performance)
        self.target_iou = torchmetrics.classification.BinaryJaccardIndex()
        self.target_f1 = torchmetrics.classification.BinaryF1Score()

    def on_train_start(self):
        # We cycle the target dataloader (held-out year)
        target_loader = self.trainer.datamodule.target_dataloader()
        self.target_iter = iter(cycle(target_loader))

    def _split_batch(self, batch):
        """Handle tuple/list or dict batches."""
        if isinstance(batch, dict):
            x = batch.get("image", batch.get("x", None))
            y = batch.get("mask", batch.get("y", None))
            return x, y
        if isinstance(batch, (list, tuple)):
            if len(batch) >= 2:
                return batch[0], batch[1]
            if len(batch) == 1:
                return batch[0], None
        return batch, None

    @torch.no_grad()
    def compute_importance(self, feat):
        logits = self.domain_head(feat)
        p_source = torch.sigmoid(logits)
        w = (1.0 - p_source) / (p_source + 1e-8)
        w = torch.sqrt(w) # Smoothing
        w = w / (w.mean() + 1e-8) # Normalize
        return torch.clamp(w, min=0.1, max=3.0)

    def training_step(self, batch, batch_idx):
        # A. Source
        x_s, y_s = self._split_batch(batch)
        # B. Target
        target_batch = next(self.target_iter)
        x_t, _ = self._split_batch(target_batch)

        if x_s.ndim == 5: x_s = x_s.flatten(1, 2)
        if x_t.ndim == 5: x_t = x_t.flatten(1, 2)

        logits_s, feat_s = self(x_s)
        logits_t, _ = self(x_t)

        # Loss 1: Weighted Source (Focus on target-like samples)
        w = self.compute_importance(feat_s)
        per_pixel_s = sigmoid_focal_loss(logits_s, y_s.unsqueeze(1), reduction="none")
        loss_s = (w * per_pixel_s.mean(dim=(1, 2, 3))).mean()

        # Loss 2: Target Entropy (Confuse the model on target data)
        p_t = torch.sigmoid(logits_t)
        entropy = -p_t * torch.log(p_t + 1e-8) - (1 - p_t) * torch.log(1 - p_t + 1e-8)
        loss_entropy = entropy.mean()

        total_loss = loss_s + (self.hparams.gamma_entropy * loss_entropy)

        # --- Monitor Feature Drift ---
        current_encoder_flat = torch.cat([p.flatten() for p in self.encoder.parameters()])
        drift = torch.norm(current_encoder_flat - self.initial_encoder_flat)

        self.log_dict({
            "train/total_loss": total_loss,
            "train/loss_entropy": loss_entropy,
            "train/encoder_drift": drift,
            "train/w_max": w.max()
        }, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        """
        Validate on Target Data with Labels.
        This gives us the truth: is adaptation helping?
        """
        x, y = batch["image"], batch["mask"]
        logits, _ = self(x)
        preds = torch.sigmoid(logits).squeeze(1)
        
        # Update metrics
        self.target_iou.update(preds, y)
        self.target_f1.update(preds, y)
        
        return {"val_loss": self.compute_loss(preds, y)}

    def on_validation_epoch_end(self):
        # Log aggregated target metrics
        self.log("val/target_iou", self.target_iou.compute(), prog_bar=True)
        self.log("val/target_f1", self.target_f1.compute(), prog_bar=True)
        self.target_iou.reset()
        self.target_f1.reset()

    def configure_optimizers(self):
        lr = self.hparams.lr
        # Paper strategy: encoder moves slower than task-specific layers
        return torch.optim.Adam([
            {'params': self.encoder.parameters(), 'lr': lr * 0.1},
            {'params': self.decoder.parameters(), 'lr': lr},
            {'params': self.seg_head.parameters(), 'lr': lr},
        ])
