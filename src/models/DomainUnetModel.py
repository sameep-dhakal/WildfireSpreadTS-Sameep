from typing import Any
from itertools import cycle
import torch.nn as nn
import torch
import segmentation_models_pytorch as smp
from .BaseModel import BaseModel
import math



# ----------------------------------------
# Gradient Reversal Layer (DANN trick)
# ----------------------------------------
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_grl):
        ctx.lambda_grl = lambda_grl
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_grl, None
    

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
    U-Net model extended for domain adaptation.
    In addition to the segmentation mask, returns encoder features.
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
        lambda_grl: float = 0.0,  
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

        # --- initialize SMP U-Net --- base segmentation model
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=n_channels,
            classes=1,
        )

        # loading the domain classifier model
        self.domain_classifier = DomainClassifier(in_channels=self.model.encoder.out_channels[-1])  # assuming resnet18 encoder 
        self.domain_loss_fn = nn.CrossEntropyLoss()

        # gradient reversal layer lambda
        self.lambda_grl = lambda_grl

        # --- domain adaptation helper ---
        # this will hold the iterator over the target dataloader (test set)
        self.target_iter = None

        print(f"✅ DomainUnetModel: loaded {encoder_name} encoder with {encoder_weights} weights")

    # ----------------------------------------------------------
    # Lightning hook: called once when training starts
    # ----------------------------------------------------------
    def on_train_start(self):
        """
        Called automatically by PyTorch Lightning before the first training step.
        Here we create an infinite iterator over the target domain (test dataloader).
        """
        dm = self.trainer.datamodule
        self.target_iter = cycle(dm.target_dataloader())
        print("✅ Target iterator initialized from target_dataloader()")

        # Optional sanity check (first batch shape)
        x_t, _ = next(self.target_iter)
        print(f"🔍 Sample target batch shape: {tuple(x_t.shape)}")

    # ----------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        doys=None,
        target: bool = False,
        return_features: bool = False,
    ):
        """
        Forward pass for domain adaptation.

        Args:
            x: input tensor (B, C, H, W) or (B, T, C, H, W)
            doys: optional (for BaseModel compatibility)
            target: if True, only extract encoder features (skip decoder)
            return_features: if True, return both mask and encoder output
        """
        # --- Flatten temporal dimension if needed ---
        if self.hparams.flatten_temporal_dimension and len(x.shape) == 5:
            x = x.flatten(start_dim=1, end_dim=2)

        # --- 1️⃣ Encoder ---
        features = self.model.encoder(x)
        encoder_output = features[-1]  # Last encoder feature map

        if target:
            # For target domain: return only encoder features
            return encoder_output
        # --- 2️⃣ Decoder + segmentation head ---
        decoder_output = self.model.decoder(*features)   # ✅ FIXED LINE
        mask = self.model.segmentation_head(decoder_output)

        in_training = hasattr(self, "trainer") and self.trainer.training

        if in_training and return_features:
            return mask, encoder_output
        else:
            return mask

    # ----------------------------------------------------------
    # Training step (source + target)
    # ----------------------------------------------------------
    def training_step(self, batch, batch_idx):
        """
        Performs one training step with source + target domain batches.
        Source: supervised segmentation loss.
        Target: encoder-only forward for domain adaptation.
        """

        # -------------------------------------------
        # 1️⃣ Segmentation loss (source only)
        # -------------------------------------------
        y_hat, y_s = self.get_pred_and_gt(batch)
        seg_loss = self.compute_loss(y_hat, y_s)

        # -------------------------------------------
        # 2️⃣ Target domain batch (unlabeled)
        # -------------------------------------------
        x_t, _ = next(self.target_iter)
        x_t = x_t.to(self.device, non_blocking=True)

        # Encoder forward pass
        f_t = self(x_t, target=True)         # (B_t, C, Hf, Wf)

        # -------------------------------------------
        # 3️⃣ Source encoder features (from same batch)
        # -------------------------------------------
        x_s, _ = batch[0], batch[1]
        if self.hparams.flatten_temporal_dimension and len(x_s.shape) == 5:
            x_s = x_s.flatten(start_dim=1, end_dim=2)
        f_s = self.model.encoder(x_s)[-1]    # (B_s, C, Hf, Wf)

        # -------------------------------------------
        # 4️⃣ Domain classifier branch
        # -------------------------------------------
        # Concatenate source + target features
        f_all = torch.cat([f_s, f_t], dim=0)

        # --------------------------------------------------
        # Compute progressive lambda (DANN schedule)
        # --------------------------------------------------
        progress = self.global_step / max(1, self.trainer.estimated_stepping_batches)
        lambda_grl = 2 / (1 + math.exp(-10 * progress)) - 1

        # Apply Gradient Reversal with scheduled lambda
        f_rev = GradientReversal.apply(f_all, lambda_grl)

        # Domain classification (2-way softmax)
        domain_logits = self.domain_classifier(f_rev)

        # Labels: 0 = source, 1 = target
        domain_labels = torch.cat([
            torch.zeros(f_s.size(0), dtype=torch.long, device=self.device),
            torch.ones(f_t.size(0), dtype=torch.long, device=self.device)
        ], dim=0)

        domain_loss = self.domain_loss_fn(domain_logits, domain_labels)

        # -------------------------------------------
        # 5️⃣ Combine losses
        # -------------------------------------------
        total_loss = seg_loss + lambda_grl * domain_loss


        self.log("train_seg_loss", seg_loss, on_step=True, on_epoch=True)
        self.log("train_domain_loss", domain_loss, on_step=True, on_epoch=True)
        self.log("train_total_loss", total_loss, on_step=True, on_epoch=True)

        # Domain classification accuracy
        with torch.no_grad():
            preds = torch.argmax(domain_logits, dim=1)
            domain_acc = (preds == domain_labels).float().mean()

        # Log for debugging/monitoring
        self.log("lambda_grl", lambda_grl, on_step=True, prog_bar=True)
        self.log("train_domain_acc", domain_acc, on_step=True, on_epoch=True)

        if batch_idx % 10 == 0:
            print(f"   λ={lambda_grl:.3f}, domain_acc={domain_acc:.3f}")

        # -------------------------------------------
        # 6️⃣ Debug info
        # -------------------------------------------
        if batch_idx % 10 == 0:
            print(f"\n🔹 [Step {batch_idx}] Source features: {tuple(f_s.shape)} | Target features: {tuple(f_t.shape)}")
            print(f"   Domain logits shape: {tuple(domain_logits.shape)}")
            probs = torch.softmax(domain_logits, dim=1)
            print(f"   Mean P(source)={probs[:,0].mean():.3f}, P(target)={probs[:,1].mean():.3f}")

        return total_loss
