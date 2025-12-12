import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss

from models.SMPModel import SMPModel
from models.DomainAdpatation.IWANStage2_WeightEstimator import (
    DomainHead3x1024,
    DomainheadCNN,
)
from ..BaseModel import BaseModel



class IWANStage3_Adaptation(BaseModel):
    """
    Stage 3: Importance-weighted segmentation training.
    - Load Stage-1 SMP UNet (encoder+decoder+head)
    - Load Stage-2 discriminator (frozen)
    - Compute importance weights = p(target|x)/mean(p(target|x))
    - Apply weighted segmentation loss in training_step
    """

    def __init__(
        self,
        encoder_name: str,
        n_channels: int,
        pos_class_weight: float,
        save_dir: str = None,
        loss_function: str = "Focal",
        stage1_ckpt: str = None,
        stage2_ckpt: str = None,
        use_doy: bool = False,
        crop_before_eval: bool = False,
        required_img_size=None,
        alpha_focal: float = 0.25,
        gamma_focal: float = 2.0,
        encoder_weights=None,
        lr: float = 1e-4,
        *args,
        **kwargs,
    ):
        super().__init__(
            n_channels=n_channels,
            flatten_temporal_dimension=True,
            pos_class_weight=pos_class_weight,
            loss_function=loss_function,
            use_doy=use_doy,
            crop_before_eval=crop_before_eval,
            required_img_size=required_img_size,
            alpha_focal=alpha_focal,
            *args,
            **kwargs,
        )
        self.lr = lr
        self.alpha_focal = alpha_focal
        self.gamma_focal = gamma_focal
        self.save_dir = save_dir  # accepted for config compatibility

        if stage1_ckpt is None:
            raise ValueError("stage1_ckpt must be provided for IWANStage3_Adaptation.")
        if stage2_ckpt is None:
            raise ValueError("stage2_ckpt must be provided for IWANStage3_Adaptation.")

        # ---------------------------------------------------------
        # Load Stage-1 segmentation model (encoder+decoder+head)
        # ---------------------------------------------------------
        base_seg: SMPModel = SMPModel.load_from_checkpoint(
            stage1_ckpt,
            encoder_name=encoder_name,
            n_channels=n_channels,
            flatten_temporal_dimension=True,
            pos_class_weight=pos_class_weight,
            encoder_weights=encoder_weights,
        )
        unet = base_seg.model
        self.encoder = unet.encoder
        self.decoder = unet.decoder
        self.seg_head = unet.segmentation_head

        # ---------------------------------------------------------
        # Load Stage-2 discriminator (frozen)
        # ---------------------------------------------------------
        state2 = torch.load(stage2_ckpt, map_location="cpu")
        feat_dim = state2.get("feat_dim", self.encoder.out_channels[-1])

        # Stage-2 exports used DomainHead3x1024; fall back to CNN if needed
        try:
            self.domain_head = DomainHead3x1024(feat_dim)
            self.domain_head.load_state_dict(state2["discriminator"], strict=True)
        except Exception:
            self.domain_head = DomainheadCNN(in_channels=feat_dim)
            self.domain_head.load_state_dict(state2["discriminator"], strict=False)

        for p in self.domain_head.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def compute_importance(self, feat):
        # feat: (B, C, H, W)
        logits = self.domain_head(feat)       # (B,)
        p_source = torch.sigmoid(logits)      # (B,)
        p_target = 1.0 - p_source  
        # Normalize first (IWAN rule)

        #softening the weights 
        w = torch.sqrt(p_target)    
        w = w / (w.mean() + 1e-8)

        return w 

    def forward(self, x, doys=None):
        if x.ndim == 5:
            x = x.flatten(1, 2)
        feats = self.encoder(x)
        dec = self.decoder(*feats)
        logits = self.seg_head(dec)
        return logits, feats[-1]

    def training_step(self, batch, batch_idx):
        # dataset returns (x, y) or (x, y, doys) depending on config
        if isinstance(batch, dict):
            x, y = batch["image"], batch["mask"]
        elif len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch

        logits, deep_feat = self(x)

        loss, w = self._weighted_loss(logits, y, deep_feat)

        # optional metrics (reuse BaseModel F1)
        self.train_f1(torch.sigmoid(logits).squeeze(1), y)
        self.log("train_f1", self.train_f1, prog_bar=True, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_weight_mean", w.mean(), prog_bar=False, on_epoch=True)
        return loss

    # def validation_step(self, batch, batch_idx):
    #     # reuse BaseModel-style loss, no weighting
    #     y_hat, y = self.get_pred_and_gt(batch)
    #     loss = self.compute_loss(y_hat, y)
    #     self.log("val_loss", loss, prog_bar=True, on_epoch=True)
    #     return loss

    # def test_step(self, batch, batch_idx):
    #     # reuse BaseModel-style test metrics (AP/F1/etc.)
    #     return super().test_step(batch, batch_idx)


    def validation_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            x, y = batch["image"], batch["mask"]
        elif len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch
        logits, _ = self(x)
        y_hat = torch.sigmoid(logits).squeeze(1)
        loss = self.compute_loss(y_hat, y)
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
        logits, _ = self(x)
        y_hat = torch.sigmoid(logits).squeeze(1)
        loss = self.compute_loss(y_hat, y)
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

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.Adam(params, lr=self.lr)

    # ============================================================
    # Helpers
    # ============================================================
    def _weighted_loss(self, logits, y, deep_feat):
        """
        Compute per-sample loss using the same per-pixel loss as BaseModel,
        then weight by IWAN p_target.
        """
        y = y.float()
        per_pixel: torch.Tensor

        if self.hparams.loss_function == "Focal":
            per_pixel = sigmoid_focal_loss(
                logits,
                y.unsqueeze(1),
                alpha=1 - self.hparams.pos_class_weight,
                gamma=self.gamma_focal,
                reduction="none",
            )
        else:
            pos_w = torch.tensor([self.hparams.pos_class_weight], device=logits.device)
            bce = nn.BCEWithLogitsLoss(pos_weight=pos_w, reduction="none")
            per_pixel = bce(logits, y.unsqueeze(1))

        per_sample = per_pixel.mean(dim=(1, 2, 3))
        w = self.compute_importance(deep_feat)
        loss = (w * per_sample).mean()
        return loss, w
