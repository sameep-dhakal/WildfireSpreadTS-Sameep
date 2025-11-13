import os
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from ..BaseModel import BaseModel


# ------------------------------------------------------------
# DOMAIN DISCRIMINATOR NETWORK
# ------------------------------------------------------------
class DomainLogitHead(nn.Module):
    """Simple 3-layer MLP: D(f) → logit."""
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # (B, C, H, W) -> (B, C, 1, 1)
            nn.Flatten(),              # (B, C)
            nn.Linear(in_channels, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),         # logit
        )

    def forward(self, x):
        return self.net(x).squeeze(1)
    

# ------------------------------------------------------------
# IWAN STAGE-2  (WEIGHT ESTIMATOR)
# ------------------------------------------------------------
class IWANStage2_WeightEstimator(BaseModel):
    """
    Stage-2 of IWAN:
        - Loads encoder Fs from Stage-1 checkpoint
        - Freezes Fs completely
        - Trains ONLY the domain head D
        - No segmentation forward is used
        - No validation in Trainer
        - Checkpoints monitor training metric score_D_epoch
    """
    def __init__(
        self,
        ckpt_dir: str,
        encoder_name: str = "resnet18",
        n_channels: int = 7,
        flatten_temporal_dimension: bool = True,
        **kwargs,
    ):
        # BaseModel needs pos_class_weight + loss, but we ignore those
        super().__init__(
            n_channels=n_channels,
            flatten_temporal_dimension=flatten_temporal_dimension,
            pos_class_weight=1.0,
            loss_function="BCE",
            use_doy=False,
            **kwargs
        )

        self.save_hyperparameters()

        # ------------------------------------------------------------
        # Load Stage-1 checkpoint
        # ------------------------------------------------------------
        if ckpt_dir is None or not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(f"❌ Checkpoint directory not found: {ckpt_dir}")

        ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
        if not ckpt_files:
            raise FileNotFoundError(f"❌ No .ckpt found inside {ckpt_dir}")

        ckpt_path = os.path.join(ckpt_dir, ckpt_files[0])
        print(f"✅ Loading encoder from Stage-1 checkpoint: {ckpt_path}")

        # Load UNet backbone
        base_model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=n_channels,
            classes=1,
        )

        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)

        # Load encoder weights only
        base_model.load_state_dict(
            {k.replace("model.", ""): v
             for k, v in sd.items()
             if "model.encoder" in k},
            strict=False,
        )

        # Freeze encoder
        self.Fs = base_model.encoder.eval()
        for p in self.Fs.parameters():
            p.requires_grad_(False)

        # Create discriminator head
        self.D = DomainLogitHead(self.Fs.out_channels[-1])
        self.bce = nn.BCEWithLogitsLoss()

        # Will store target iterator
        self.target_iter = None

    # ------------------------------------------------------------
    # Forward through encoder + domain head
    # ------------------------------------------------------------
    def forward_D(self, x_s, x_t):

        # flatten temporal dimension (B, T, C, H, W) → (B, T*C, H, W)
        if self.hparams.flatten_temporal_dimension and x_s.ndim == 5:
            x_s = x_s.flatten(start_dim=1, end_dim=2)
        if self.hparams.flatten_temporal_dimension and x_t.ndim == 5:
            x_t = x_t.flatten(start_dim=1, end_dim=2)

        with torch.no_grad():
            f_s = self.Fs(x_s)[-1]   # final encoder feature map
            f_t = self.Fs(x_t)[-1]

        log_s = self.D(f_s)
        log_t = self.D(f_t)

        y_s = torch.ones_like(log_s)
        y_t = torch.zeros_like(log_t)

        loss = 0.5 * self.bce(log_s, y_s) + 0.5 * self.bce(log_t, y_t)

        with torch.no_grad():
            p_s = torch.sigmoid(log_s)
            p_t = torch.sigmoid(log_t)
            acc = 0.5 * ((p_s > 0.5).float().mean() + (p_t < 0.5).float().mean())

        # For checkpoint monitoring
        score = loss.detach()

        return loss, acc, score

    # ------------------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------------------
    def on_train_start(self):
        dm = self.trainer.datamodule
        self.target_iter = iter(dm.target_dataloader())
        print("🟢 Stage-2: target dataloader iterator initialized.")

    def training_step(self, batch, _):
        x_s, _ = batch

        try:
            x_t, _ = next(self.target_iter)
        except StopIteration:
            self.target_iter = iter(self.trainer.datamodule.target_dataloader())
            x_t, _ = next(self.target_iter)

        x_s = x_s.to(self.device)
        x_t = x_t.to(self.device)

        loss, acc, score = self.forward_D(x_s, x_t)

        self.log("train_loss_D", loss, prog_bar=True, on_epoch=True)
        self.log("train_domain_acc", acc, prog_bar=True, on_epoch=True)
        self.log("score_D_epoch", score, prog_bar=False, on_epoch=True)

        return loss

    # ------------------------------------------------------------
    # Disable validation logic completely
    # ------------------------------------------------------------
    def validation_step(self, *args, **kwargs):
        return None

    # ------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------
    def configure_optimizers(self):
        # Only train domain head D
        return torch.optim.Adam(self.D.parameters(), lr=1e-4)

    # ------------------------------------------------------------
    # Forward_OVERRIDE (never used for segmentation)
    # ------------------------------------------------------------
    def forward(self, x):
        # flatten if needed
        if self.hparams.flatten_temporal_dimension and x.ndim == 5:
            x = x.flatten(start_dim=1, end_dim=2)

        with torch.no_grad():
            f = self.Fs(x)[-1]

        return self.D(f)
