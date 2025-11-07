from typing import Any
from itertools import cycle
import math

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from .BaseModel import BaseModel
from torchvision.ops import sigmoid_focal_loss


class DomainClassifier(nn.Module):
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
            nn.Linear(1024, 2)        # 2 classes: source / target
        )

    def forward(self, x):
        return self.net(x)


class DomainUnetModel(BaseModel):
    """
    U-Net model extended for *importance-weighted* domain adaptation.

    - Encoder/decoder: SMP U-Net (label predictor)
    - DomainClassifier: estimates p(target | features)
    - NO GRL, NO adversarial gradient
    - We use p(target | source sample) as an importance weight
      on the segmentation loss.
    """

    def __init__(
        self,
        encoder_name: str = "resnet18",
        encoder_weights: str = "imagenet",
        n_channels: int = 7,
        flatten_temporal_dimension: bool = True,
        pos_class_weight: float = 1.0,
        loss_function: str = "Focal",
        use_doy: bool = False,
        crop_before_eval: bool = False,
        required_img_size=None,
        alpha_focal=None,
        f1_threshold=None,
        lambda_grl: float = 0.1,  # 🔁 now used as *domain_loss_weight*
        **kwargs: Any
    ):
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
            **kwargs
        )

        self.save_hyperparameters()

        encoder_weights = encoder_weights if encoder_weights != "none" else None

        # --- base segmentation model ---
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=n_channels,
            classes=1,
        )

        # --- domain classifier for importance weighting ---
        self.domain_classifier = DomainClassifier(
            in_channels=self.model.encoder.out_channels[-1]
        )
        self.domain_loss_fn = nn.CrossEntropyLoss()

        # 🔁 lambda_grl is no longer GRL strength;
        #     we reuse it as the *weight on domain_loss*
        self.domain_loss_weight = lambda_grl

        # target iterator (lazy)
        self.target_iter = None

        print(f"✅ DomainUnetModel: loaded {encoder_name} encoder with {encoder_weights} weights")

    # ----------------------------------------------------------
    # Hook: called once when training starts
    # ----------------------------------------------------------
    def on_train_start(self):
        """
        Delay target dataloader initialization until first training batch.
        This avoids long blocking during setup.
        """
        self.target_iter = None
        print("🕓 Target iterator will be created lazily on first training batch.")

    # ----------------------------------------------------------
    # Forward: regular segmentation
    # ----------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        doys=None,
        target: bool = False,
        return_features: bool = False,
    ):
        """
        Args:
            x: (B, C, H, W) or (B, T, C, H, W)
            target: if True → return encoder features only
            return_features: if True → (mask, encoder_output) during training
        """
        # flatten time if needed (same as BaseModel)
        if self.hparams.flatten_temporal_dimension and len(x.shape) == 5:
            x = x.flatten(start_dim=1, end_dim=2)

        # 1) encoder
        features = self.model.encoder(x)
        encoder_output = features[-1]  # (B, C, Hf, Wf)

        if target:
            return encoder_output

        # 2) decoder + segmentation head
        decoder_output = self.model.decoder(*features)
        mask = self.model.segmentation_head(decoder_output)

        in_training = hasattr(self, "trainer") and self.trainer.training
        if in_training and return_features:
            return mask, encoder_output
        else:
            return mask

    # ----------------------------------------------------------
    # Helper: per-sample segmentation loss (for importance weights)
    # ----------------------------------------------------------
    def compute_seg_loss_per_sample(self, y_hat: torch.Tensor, y: torch.Tensor):
        """
        Returns a loss vector of shape (B,) to allow per-sample weighting.
        Supports Focal and BCE. For other losses, falls back to uniform.
        """
        if self.hparams.loss_function == "Focal":
            # shape: same as y_hat
            loss = sigmoid_focal_loss(
                y_hat,
                y.float(),
                alpha=1 - self.hparams.pos_class_weight,
                gamma=2,
                reduction="none",
            )
        elif self.hparams.loss_function == "BCE":
            bce = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(
                    [self.hparams.pos_class_weight],
                    device=y_hat.device,
                    dtype=torch.float32
                ),
                reduction="none",
            )
            loss = bce(y_hat, y.float())
        else:
            # fallback: use scalar loss and broadcast
            scalar_loss = self.compute_loss(y_hat, y)
            return scalar_loss.repeat(y_hat.size(0))

        # y_hat from BaseModel.get_pred_and_gt is (B, H, W)
        # or (B, 1, H, W) depending on path; reduce to (B,)
        if loss.dim() == 4:
            loss = loss.mean(dim=(1, 2, 3))
        elif loss.dim() == 3:
            loss = loss.mean(dim=(1, 2))
        elif loss.dim() == 2:
            loss = loss.mean(dim=1)

        return loss  # (B,)

    # ----------------------------------------------------------
    # Training step: importance weighting
    # ----------------------------------------------------------
    def training_step(self, batch, batch_idx):
        """
        1. Compute segmentation outputs on source batch.
        2. Get features for source and target.
        3. Train domain classifier on *detached* features.
        4. Use p(target|source) as importance weights on segmentation loss.
        """

        # Lazily init target iterator
        if self.target_iter is None:
            dm = self.trainer.datamodule
            self.target_iter = cycle(dm.target_dataloader())
            print("✅ Target iterator initialized from target_dataloader() (lazy mode)")

        # -------------------------------------------
        # 1️⃣ Segmentation on source
        # -------------------------------------------
        y_hat, y_s = self.get_pred_and_gt(batch)           # uses self.forward internally
        seg_loss_per_sample = self.compute_seg_loss_per_sample(y_hat, y_s)  # (B,)

        # -------------------------------------------
        # 2️⃣ Target domain batch
        # -------------------------------------------
        x_t, _ = next(self.target_iter)
        x_t = x_t.to(self.device, non_blocking=True)

        # Encoder features (NO GRL)
        f_t = self(x_t, target=True)         # (B_t, C, Hf, Wf)

        # -------------------------------------------
        # 3️⃣ Source encoder features (same batch)
        # -------------------------------------------
        x_s, _ = batch[0], batch[1]
        if self.hparams.flatten_temporal_dimension and len(x_s.shape) == 5:
            x_s = x_s.flatten(start_dim=1, end_dim=2)
        f_s = self.model.encoder(x_s)[-1]    # (B_s, C, Hf, Wf)

        # -------------------------------------------
        # 4️⃣ Train domain classifier on DETACHED features
        # -------------------------------------------
        f_s_det = f_s.detach()
        f_t_det = f_t.detach()

        logits_s = self.domain_classifier(f_s_det)
        logits_t = self.domain_classifier(f_t_det)

        domain_logits = torch.cat([logits_s, logits_t], dim=0)
        domain_labels = torch.cat([
            torch.zeros(f_s_det.size(0), dtype=torch.long, device=self.device),
            torch.ones(f_t_det.size(0), dtype=torch.long, device=self.device)
        ], dim=0)

        domain_loss = self.domain_loss_fn(domain_logits, domain_labels)

        # -------------------------------------------
        # 5️⃣ Compute importance weights for source samples
        # -------------------------------------------
        with torch.no_grad():
            probs_s = torch.softmax(logits_s, dim=1)  # (B_s, 2)
            p_target = probs_s[:, 1]                  # p(d=target | source sample)
            # Normalize and clip to avoid extremes
            w = p_target / (p_target.mean() + 1e-6)
            w = torch.clamp(w, 0.1, 10.0)

        # Weighted segmentation loss
        weighted_seg_loss = (w * seg_loss_per_sample).mean()

        # -------------------------------------------
        # 6️⃣ Combine losses
        #     - weighted_seg_loss updates encoder + decoder
        #     - domain_loss updates *only* domain_classifier (features are detached)
        # -------------------------------------------
        total_loss = weighted_seg_loss + self.domain_loss_weight * domain_loss

        # Logging
        self.log("train_seg_loss", weighted_seg_loss, on_step=True, on_epoch=True)
        self.log("train_domain_loss", domain_loss, on_step=True, on_epoch=True)
        self.log("train_total_loss", total_loss, on_step=True, on_epoch=True)
        self.log("mean_importance_weight", w.mean(), on_step=True, on_epoch=True)

        # Domain classification accuracy (just for monitoring)
        with torch.no_grad():
            preds = torch.argmax(domain_logits, dim=1)
            domain_acc = (preds == domain_labels).float().mean()
        self.log("train_domain_acc", domain_acc, on_step=True, on_epoch=True)

        # Debug prints every 10 steps
        if batch_idx % 10 == 0:
            print(f"\n🔹 [Step {batch_idx}] Source features: {tuple(f_s.shape)} | Target features: {tuple(f_t.shape)}")
            print(f"   Domain logits shape: {tuple(domain_logits.shape)}")
            print(f"   Domain acc={domain_acc:.3f}, mean w={w.mean():.3f}")
            print(f"   Mean P(source)={probs_s[:,0].mean():.3f}, P(target)={probs_s[:,1].mean():.3f}")

        return total_loss
