import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from ..BaseModel import BaseModel


# ============================================================
# DOMAIN HEAD (DISCRIMINATOR)
# ============================================================
class DomainLogitHead(nn.Module):
    """Small stable MLP: pooled feature map → logit."""
    def __init__(self, in_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # (B, C, H, W) → (B, C, 1, 1)
            nn.Flatten(),              # (B, C)
            nn.Linear(in_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)
        

# ============================================================
# IWAN STAGE 2 WEIGHT ESTIMATOR
# ============================================================
class IWANStage2_WeightEstimator(BaseModel):
    """
    IWAN Stage-2:
        - Load encoder Fs from Stage-1 (UNet)
        - Freeze encoder (including disabling BN updates)
        - Train only a small domain classifier D
        - No segmentation
        - No validation
        - Checkpoint monitors: score_D_epoch (min)
    """

    def __init__(
        self,
        encoder_name: str = "resnet18",
        encoder_weights: Optional[str] = None,
        in_channels: Optional[int] = None,
        n_channels: int = 7,
        flatten_temporal_dimension: bool = True,
        pos_class_weight: float = 1.0,
        loss_function: str = "BCE",
        use_doy: bool = False,
        crop_before_eval: bool = False,
        required_img_size: Optional[Tuple[int, int]] = None,
        alpha_focal: Optional[float] = None,
        f1_threshold: Optional[float] = None,
        ckpt_dir: str = None,
        **kwargs,
    ):
        # allow in_channels override
        if in_channels is not None:
            n_channels = in_channels

        # initialize BaseModel (Lightning compatible)
        super().__init__(
            n_channels=n_channels,
            flatten_temporal_dimension=flatten_temporal_dimension,
            pos_class_weight=pos_class_weight,
            loss_function=loss_function,
            use_doy=use_doy,
            crop_before_eval=crop_before_eval,
            required_img_size=required_img_size,
            alpha_focal=alpha_focal,
            f1_threshold=f1_threshold,
            **kwargs,
        )

        # save all hparams
        self.save_hyperparameters()

        # ---------------------------------------------
        # LOAD STAGE-1 CHECKPOINT
        # ---------------------------------------------
        if ckpt_dir is None or not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(f"❌ Stage-1 ckpt directory not found: {ckpt_dir}")

        ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
        if not ckpt_files:
            raise FileNotFoundError(f"❌ No .ckpt files found in {ckpt_dir}")

        ckpt_path = os.path.join(ckpt_dir, ckpt_files[0])
        print(f"✅ Loading Stage-1 encoder from: {ckpt_path}")

        stage1 = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=n_channels,
            classes=1,
        )

        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)

        # load encoder weights only
        stage1.load_state_dict(
            {k.replace("model.", ""): v for k, v in sd.items() if "encoder" in k},
            strict=False
        )

        # ---------------------------------------------
        # CORRECT MODULE REGISTRATION
        # ---------------------------------------------
        self.Fs = stage1.encoder            # register module
        self.Fs.eval()                      # set eval mode

        # Fully freeze encoder parameters
        for p in self.Fs.parameters():
            p.requires_grad_(False)

        # Disable BN update
        for m in self.Fs.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None

        # ---------------------------------------------
        # DOMAIN DISCRIMINATOR
        # ---------------------------------------------
        feat_dim = self.Fs.out_channels[-1]
        self.D = DomainLogitHead(feat_dim)
        self.bce = nn.BCEWithLogitsLoss()

        # iterator for target batches
        self.target_iter = None

    # ============================================================
    # DISCRIMINATOR FORWARD
    # ============================================================
    def forward_D(self, x_s, x_t):
        # flatten T dimension if needed
        if self.hparams.flatten_temporal_dimension and x_s.ndim == 5:
            x_s = x_s.flatten(1, 2)
        if self.hparams.flatten_temporal_dimension and x_t.ndim == 5:
            x_t = x_t.flatten(1, 2)

        with torch.no_grad():                      # encoder NEVER updates
            f_s = self.Fs(x_s)[-1]
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

        score = loss.detach()

        return loss, acc, score

    # ============================================================
    # TRAINING LOOP
    # ============================================================
    def on_train_start(self):
        dm = self.trainer.datamodule
        self.target_iter = iter(dm.target_dataloader())
        print("🟢 Stage-2: initialized target iterator")

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

        self.log("train_loss_D", loss, on_epoch=True, prog_bar=True)
        self.log("train_domain_acc", acc, on_epoch=True, prog_bar=True)
        self.log("score_D", score, on_epoch=True, prog_bar=False)

        return loss

    # ============================================================
    # NO VALIDATION
    # ============================================================
    def validation_step(self, *args, **kwargs):
        return None

    # ============================================================
    # CORRECT OPTIMIZER (NOW NOT OVERRIDDEN BY CLI)
    # ============================================================
    def configure_optimizers(self):
        return torch.optim.Adam(self.D.parameters(), lr=5e-5)

    # ============================================================
    # FORWARD (NEVER USED)
    # ============================================================
    def forward(self, x):
        if self.hparams.flatten_temporal_dimension and x.ndim == 5:
            x = x.flatten(1, 2)

        with torch.no_grad():
            f = self.Fs(x)[-1]

        return self.D(f)
