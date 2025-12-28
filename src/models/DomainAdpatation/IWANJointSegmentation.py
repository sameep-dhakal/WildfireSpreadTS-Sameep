"""
IWAN-style joint domain adaptation for segmentation using the SMP UNet backbone.
- Uses the same data interface as SMPModel/BaseModel (FireSpreadDataModule train_dataloader for source, target_dataloader for target).
- Implements the two-encoder IWAN objective: frozen Fs for weights and source branch, trainable Ft with GRL for target branch, optional target entropy.
"""
import copy

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.autograd import Function

from models.SMPModel import SMPModel
from models.DomainAdpatation.IWANStage2_WeightEstimator import DomainHead3x1024, DomainheadCNN
from ..BaseModel import BaseModel


class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd: float):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd: float = 1.0):
    return _GradReverse.apply(x, lambd)


class IWANJointSegmentation(BaseModel):
    """
    Two-encoder IWAN-style segmentation matching the paper’s staged setup:
      - Frozen source encoder Fs (from Stage-1) → z_s feeds D for weights and D0 source branch
      - Trainable target encoder Ft (init from Stage-1) → z_t feeds D/D0 target branch and seg outputs
      - D trains on detached features (no GRL, no Ft gradients) to estimate w(z_s)
      - D0 plays weighted adversarial game with GRL on Ft; source branch weighted by normalized w(z_s)
      - Optional target entropy regularization; segmentation head/decoder are fixed (as classifier C)
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
        alpha_focal: float = 0.25,
        f1_threshold=None,
        stage1_ckpt: str = None,
        domain_head: str = "cnn",  # default: spatial CNN for satellite features; "iwan" -> 2-layer 1024 MLP (reference), "mlp" -> 3x1024
        lambda_D: float = 1.5,
        lambda_D0: float = 1.0,
        lambda_grl: float = 1.0,
        entropy_weight: float = 0.01,
        lr_head_scale: float = 0.1,  # lower LR for decoder/seg head to co-adapt gently
        lr: float = 1e-4,
        *args,
        **kwargs,
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
            *args,
            **kwargs,
        )
        self.save_hyperparameters()

        self.lambda_D = lambda_D
        self.lambda_D0 = lambda_D0
        self.lambda_grl = lambda_grl
        self.entropy_weight = entropy_weight
        self.lr = lr
        self.lr_head_scale = lr_head_scale
        # We control optimization manually to separate D vs (Ft,D0) steps
        self.automatic_optimization = False
        self._sigmoid_alpha = 1.0  # schedule parameter for lambda_D0 (paper-style)
        self._lambda_upper = lambda_D0

        # ------------------------------------------------------------------
        # Segmentation backbone (must come from Stage-1 to build Fs and Ft)
        # ------------------------------------------------------------------
        if not stage1_ckpt:
            raise ValueError("stage1_ckpt is required to build frozen Fs and trainable Ft for IWAN.")

        base: SMPModel = SMPModel.load_from_checkpoint(
            stage1_ckpt,
            encoder_name=encoder_name,
            n_channels=n_channels,
            flatten_temporal_dimension=flatten_temporal_dimension,
            pos_class_weight=pos_class_weight,
            encoder_weights=None if encoder_weights == "none" else encoder_weights,
        )
        unet = base.model

        # Trainable Ft
        self.encoder_t = unet.encoder
        self.decoder = unet.decoder
        self.seg_head = unet.segmentation_head

        # Frozen Fs (copy of encoder_t)
        self.encoder_s = copy.deepcopy(self.encoder_t)
        for p in self.encoder_s.parameters():
            p.requires_grad_(False)
        self.encoder_s.eval()

        # ------------------------------------------------------------------
        # Domain discriminators D and D0
        # ------------------------------------------------------------------
        self.domain_D = self._make_domain_head(domain_head)
        self.domain_D0 = self._make_domain_head(domain_head)
        self.domain_bce = nn.BCEWithLogitsLoss(reduction="none")

        self.target_iter = None

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def _make_domain_head(self, name: str):
        in_ch = self.encoder_t.out_channels[-1]
        if name.lower() == "iwan":
            return DomainHeadIWAN(in_channels=in_ch)
        if name.lower() == "cnn":
            return DomainheadCNN(in_channels=in_ch)
        return DomainHead3x1024(in_channels=in_ch)

    def _flatten_input(self, x):
        if self.hparams.flatten_temporal_dimension and x.ndim == 5:
            x = x.flatten(1, 2)
        return x

    def forward(self, x, doys=None):
        x = self._flatten_input(x)
        feats = self.encoder_t(x)
        dec = self.decoder(*feats)
        logits = self.seg_head(dec)
        return logits, feats[-1]

    # ----------------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------------
    def on_train_start(self):
        dm = self.trainer.datamodule
        if not hasattr(dm, "target_dataloader"):
            raise RuntimeError("FireSpreadDataModule must implement target_dataloader for IWAN joint training.")
        self.target_iter = iter(dm.target_dataloader())

    def _next_target_batch(self):
        try:
            x_t, _ = next(self.target_iter)
        except StopIteration:
            self.target_iter = iter(self.trainer.datamodule.target_dataloader())
            x_t, _ = next(self.target_iter)
        return x_t

    def training_step(self, batch, batch_idx):
        opt_D, opt_main = self.optimizers()

        # source batch
        if isinstance(batch, dict):
            x_s, y_s = batch["image"], batch["mask"]
        elif len(batch) == 3:
            x_s, y_s, _ = batch
        else:
            x_s, y_s = batch

        x_t = self._next_target_batch()

        x_s = x_s.to(self.device)
        y_s = y_s.to(self.device)
        x_t = x_t.to(self.device)

        # frozen Fs for source weights
        zs = self.encoder_s(self._flatten_input(x_s))[-1]

        # trainable Ft for source (seg anchor) and target (adaptation)
        feats_s_t = self.encoder_t(self._flatten_input(x_s))
        feats_t_t = self.encoder_t(self._flatten_input(x_t))
        feat_s = feats_s_t[-1]
        feat_t = feats_t_t[-1]

        # forward through decoder/seg head (seg anchor + entropy/metrics)
        logits_s = self.seg_head(self.decoder(*feats_s_t))
        logits_t = self.seg_head(self.decoder(*feats_t_t))

        # Domain loss for D (detached features)
        # Eq (2): L_D = 0.5 * (ℓ_bce(D(f_s),1) + ℓ_bce(D(f_t),0))
        log_s_D = self.domain_D(zs.detach())
        log_t_D = self.domain_D(feat_t.detach())
        ones_s = torch.ones_like(log_s_D)
        zeros_t = torch.zeros_like(log_t_D)
        loss_D = 0.5 * (
            self.domain_bce(log_s_D, ones_s).mean() +
            self.domain_bce(log_t_D, zeros_t).mean()
        )
        # Step D only
        opt_D.zero_grad()
        self.manual_backward(self.lambda_D * loss_D)
        opt_D.step()

        # IWAN weights: probability of target for source samples
        # Eq (3): w_s = 1 - sigmoid(D(f_s)) = p_t
        with torch.no_grad():
            tilde_w = 1.0 - torch.sigmoid(log_s_D)
            w_s = tilde_w / (tilde_w.mean() + 1e-8)  # Eq (8)/(11) normalization

        # Domain loss for D0 with GRL, weighted by w_s
        # Eq (4): L_D0 = 0.5*(E[w_s * ℓ_bce(D0(GRL(f_s)),1)] + E[ℓ_bce(D0(GRL(f_t)),0)])
        log_s_D0 = self.domain_D0(zs.detach())  # Fs branch, no GRL, no Ft grad
        log_t_D0 = self.domain_D0(grad_reverse(feat_t, self.lambda_grl))

        loss_s_D0 = (w_s * self.domain_bce(log_s_D0, ones_s)).mean()
        loss_t_D0 = self.domain_bce(log_t_D0, zeros_t).mean()
        loss_D0 = 0.5 * (loss_s_D0 + loss_t_D0)

        # Source segmentation anchor to preserve task performance
        seg_loss = self.compute_loss(logits_s.squeeze(1), y_s)

        # Optional entropy regularization on target predictions
        ent_loss = 0.0
        if self.entropy_weight > 0:
            p_t = torch.sigmoid(logits_t)
            ent = -(p_t * torch.log(p_t + 1e-8) + (1 - p_t) * torch.log(1 - p_t + 1e-8))
            ent_loss = ent.mean()

        # Paper-style schedule for lambda_D0: p in [0,1], lambda = 2*u/(1+exp(-alpha*p)) - u
        progress = min(1.0, (self.trainer.global_step + 1) / max(1, self.trainer.estimated_stepping_batches))
        lambda_sched = 2 * self._lambda_upper / (1.0 + torch.exp(torch.tensor(-self._sigmoid_alpha * progress, device=self.device))) - self._lambda_upper

        # Eq (5/14) with segmentation anchor: L = L_seg + λ_D0(t) L_D0 + λ_ent H(p_t); L_D already stepped separately
        main_loss = seg_loss + lambda_sched * loss_D0 + self.entropy_weight * ent_loss

        opt_main.zero_grad()
        self.manual_backward(main_loss)
        opt_main.step()

        self.log("train_loss", main_loss, prog_bar=True, on_epoch=True)
        self.log("train_seg_loss", seg_loss, prog_bar=False, on_epoch=True)
        self.log("train_D_loss", loss_D, prog_bar=False, on_epoch=True)
        self.log("train_D0_loss", loss_D0, prog_bar=False, on_epoch=True)
        self.log("lambda_D0_eff", lambda_sched, prog_bar=False, on_epoch=True)
        if self.entropy_weight > 0:
            self.log("train_ent_loss", ent_loss, prog_bar=False, on_epoch=True)

        # Source segmentation metric to monitor preservation
        self.train_f1(torch.sigmoid(logits_s).squeeze(1), y_s)
        self.log("train_f1", self.train_f1, prog_bar=True, on_epoch=True)

        return {"loss": main_loss.detach()}

    # ----------------------------------------------------------------------
    # Validation / Test (standard segmentation)
    # ----------------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            x, y = batch["image"], batch["mask"]
        elif len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch
        logits, _ = self.forward(x)
        y_hat = torch.sigmoid(logits).squeeze(1)
        loss = self.compute_loss(logits.squeeze(1), y)
        self.val_avg_precision(y_hat, y)
        self.val_f1(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_avg_precision", self.val_avg_precision, prog_bar=True, on_epoch=True)
        self.log("val_f1", self.val_f1, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            x, y = batch["image"], batch["mask"]
        elif len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch
        logits, _ = self.forward(x)
        y_hat = torch.sigmoid(logits).squeeze(1)
        loss = self.compute_loss(logits.squeeze(1), y)
        self.test_f1(y_hat, y)
        self.test_avg_precision(y_hat, y)
        self.test_precision(y_hat, y)
        self.test_recall(y_hat, y)
        self.test_iou(y_hat, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log_dict(
            {
                "test_f1": self.test_f1,
                "test_AP": self.test_avg_precision,
                "test_precision": self.test_precision,
                "test_recall": self.test_recall,
                "test_iou": self.test_iou,
            }
        )
        return loss

    # ----------------------------------------------------------------------
    def configure_optimizers(self):
        opt_D = torch.optim.Adam(self.domain_D.parameters(), lr=self.lr)
        opt_main = torch.optim.Adam(
            [
                {"params": self.encoder_t.parameters(), "lr": self.lr},
                {"params": list(self.decoder.parameters()) + list(self.seg_head.parameters()), "lr": self.lr * self.lr_head_scale},
                {"params": self.domain_D0.parameters(), "lr": self.lr},
            ],
        )
        return [opt_D, opt_main]


class DomainHeadIWAN(nn.Module):
    """
    Two-layer MLP domain discriminator mirroring the reference IWAN/DANN setup:
    AdaptiveAvgPool -> Flatten -> Linear(C,1024) -> ReLU -> Linear(1024,1)
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)
