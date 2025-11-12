import os
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from .BaseModel import BaseModel


# ------------------------------------------------------------
#  DOMAIN DISCRIMINATOR NETWORK
# ------------------------------------------------------------
class DomainLogitHead(nn.Module):
    """3-layer MLP domain classifier D(f) = p(source|feature)."""
    def __init__(self, in_channels=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B, C, H, W) → (B, C, 1, 1)
            nn.Flatten(),             # (B, C)
            nn.Linear(in_channels, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),       # Binary domain logit
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


# ------------------------------------------------------------
#  STAGE 2: DOMAIN WEIGHT ESTIMATOR (IWAN)
# ------------------------------------------------------------
class IWANStage2_WeightEstimator(BaseModel):
    """
    Stage 2 of IWAN (Zhang et al., CVPR 2018)
    -----------------------------------------
    - Fs (source encoder) is frozen from Stage 1.
    - D learns to discriminate source vs target features.
    - Validation loss is used for robust checkpointing.
    """

    def __init__(self, ckpt_dir: str, encoder_name="resnet18", n_channels=7, **kwargs):
        super().__init__(n_channels=n_channels, **kwargs)
        self.save_hyperparameters()

        # -------------------------
        # Load pretrained source encoder Fs
        # -------------------------
        ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
        if not ckpt_files:
            raise FileNotFoundError(f"No .ckpt file found in {ckpt_dir}")
        ckpt_path = os.path.join(ckpt_dir, ckpt_files[0])
        print(f"✅ Loading frozen encoder from: {ckpt_path}")

        base_model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=n_channels,
            classes=1,
        )
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        base_model.load_state_dict(
            {k.replace("model.", ""): v for k, v in state_dict.items() if "model." in k},
            strict=False,
        )

        # Freeze encoder weights
        self.Fs = base_model.encoder.eval()
        for p in self.Fs.parameters():
            p.requires_grad_(False)

        # Domain discriminator D
        self.D = DomainLogitHead(self.Fs.out_channels[-1])
        self.bce = nn.BCEWithLogitsLoss()

        # Lazy iterator for target domain
        self.target_iter = None

    # ------------------------------------------------------------
    # HELPER: Forward through frozen encoder + domain head
    # ------------------------------------------------------------
    def forward_D(self, x_s, x_t):
        with torch.no_grad():
            f_s = self.Fs(x_s)[-1]
            f_t = self.Fs(x_t)[-1]

        log_s, log_t = self.D(f_s), self.D(f_t)
        y_s, y_t = torch.ones_like(log_s), torch.zeros_like(log_t)
        loss = 0.5 * self.bce(log_s, y_s) + 0.5 * self.bce(log_t, y_t)

        with torch.no_grad():
            p_s, p_t = torch.sigmoid(log_s), torch.sigmoid(log_t)
            acc = 0.5 * ((p_s > 0.5).float().mean() + (p_t < 0.5).float().mean())

        return loss, acc

    # ------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------
    def on_train_start(self):
        """Prepare a lazy target iterator to avoid blocking at setup."""
        dm = self.trainer.datamodule
        self.target_iter = iter(dm.target_dataloader())
        print("🟢 Initialized target iterator for Stage 2 training.")

    def training_step(self, batch, _):
        x_s, _ = batch[0], batch[1]

        # get a target batch
        try:
            x_t, _ = next(self.target_iter)
        except StopIteration:
            self.target_iter = iter(self.trainer.datamodule.target_dataloader())
            x_t, _ = next(self.target_iter)

        x_t = x_t.to(self.device, non_blocking=True)
        x_s = x_s.to(self.device, non_blocking=True)

        loss, acc = self.forward_D(x_s, x_t)
        self.log("train_loss_D", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_domain_acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    # ------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------
    def validation_step(self, batch, _):
        """Compute D’s validation loss on held-out source/target samples."""
        x_s, _ = batch[0], batch[1]

        try:
            x_t, _ = next(self.target_iter)
        except StopIteration:
            self.target_iter = iter(self.trainer.datamodule.target_dataloader())
            x_t, _ = next(self.target_iter)

        x_t = x_t.to(self.device, non_blocking=True)
        x_s = x_s.to(self.device, non_blocking=True)

        val_loss, val_acc = self.forward_D(x_s, x_t)
        self.log("val_loss_D", val_loss, prog_bar=True, on_epoch=True)
        self.log("val_domain_acc", val_acc, prog_bar=True, on_epoch=True)
        return val_loss

    # ------------------------------------------------------------
    # OPTIMIZER
    # ------------------------------------------------------------
    def configure_optimizers(self):
        return torch.optim.Adam(self.D.parameters(), lr=1e-4)
