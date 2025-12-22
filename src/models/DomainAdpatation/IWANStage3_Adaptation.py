import torch
import torch.nn as nn
from torch.autograd import Function
from itertools import cycle
import torchmetrics
import copy

from models.SMPModel import SMPModel
from models.DomainAdpatation.IWANStage2_WeightEstimator import DomainHead3x1024
from ..BaseModel import BaseModel

# --- GRL Implementation: Paper Section 3.2 ---
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Reverses gradient by -1 for minimax optimization
        return grad_output.neg() * ctx.alpha, None

class GRL(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

class IWANStage3_Adaptation(BaseModel):
    def __init__(
        self,
        encoder_name: str,
        n_channels: int,
        pos_class_weight: float,
        save_dir: str = "/develop/results/",
        stage1_ckpt: str = None,
        stage2_ckpt: str = None,
        gamma_entropy: float = 0.1,        # γ in Eq 14 (entropy on target)
        lambda_adv: float = 0.5,           # final λ in Eq 14 (adversarial weight)
        lambda_adv_warmup_epochs: int = 3, # linearly ramp λ over these epochs
        lr: float = 1e-4,
        lr_head_scale: float = 0.5,        # smaller LR for decoder/seg head to keep them stable
        **kwargs,
    ):
        super().__init__(n_channels=n_channels, pos_class_weight=pos_class_weight, **kwargs)
        self.save_hyperparameters(ignore=["stage1_ckpt", "stage2_ckpt"])

        # 1. LOAD ARCHITECTURES (Unshared Feature Extractors)
        base_seg = SMPModel.load_from_checkpoint(stage1_ckpt, encoder_name=encoder_name)
        
        # Fs: Source Feature Extractor (FROZEN)
        self.encoder_s = base_seg.model.encoder
        for p in self.encoder_s.parameters(): p.requires_grad = False
        
        # Ft: Target Feature Extractor (TRAINABLE, initialized from Fs)
        # Use deepcopy to ensure they are physically separate in GPU memory
        self.encoder_t = copy.deepcopy(base_seg.model.encoder) 
        
        # C: Classifier/Decoder (FROZEN to preserve task expertise)
        self.decoder = base_seg.model.decoder
        self.seg_head = base_seg.model.segmentation_head
        for p in self.decoder.parameters(): p.requires_grad = True
        for p in self.seg_head.parameters(): p.requires_grad = True

        # 2. DOMAIN DISCRIMINATORS
        feat_dim = self.encoder_s.out_channels[-1]
        
        # D: Oracle from Stage 2 (FROZEN) to get importance weight w(z)
        state2 = torch.load(stage2_ckpt, map_location="cpu")
        self.domain_oracle = DomainHead3x1024(feat_dim)
        self.domain_oracle.load_state_dict(state2["discriminator"])
        for p in self.domain_oracle.parameters(): p.requires_grad = False
        
        # D0: SECOND DOMAIN CLASSIFIER (TRAINABLE)
        self.domain_adv = DomainHead3x1024(feat_dim)
        self.grl = GRL(alpha=1.0)

        # 3. METRICS (Re-initializing for the target year)
        self.target_iou = torchmetrics.classification.BinaryJaccardIndex()

    def on_train_start(self):
        target_loader = self.trainer.datamodule.target_dataloader()
        self.target_iter = iter(cycle(target_loader))

    def _split_batch(self, batch):
        if isinstance(batch, dict):
            return batch.get("image"), batch.get("mask")
        return batch[0], batch[1]

    # @torch.no_grad()
    # def compute_importance(self, feat):
    #     """ Eq 7 & 8: Importance weights w(z) """
    #     d_out = torch.sigmoid(self.domain_oracle(feat))
    #     w_tilde = (1.0 - d_out) / (d_out + 1e-8)
    #     # Normalization as per Eq 8
    #     return w_tilde / (w_tilde.mean() + 1e-8)


    @torch.no_grad()
    def compute_importance(self, feat):
        """ 
        Enhanced Eq 7 & 8: Importance weights with smoothing and clamping.
        """
        # 1. Get raw probability from first discriminator D [cite: 117]
        d_out = torch.sigmoid(self.domain_oracle(feat)) 
        
        # 2. Compute raw weight ratio (Eq 7) [cite: 125]
        w_tilde = (1.0 - d_out) / (d_out + 1e-8)
        
        # 3. Apply Square Root Smoothing
        # This reduces the variance of the weights, making training more stable 
        w_tilde = torch.sqrt(w_tilde)
        
        # 4. Normalize weights so the expected value is 1 (Eq 8) [cite: 132, 147]
        w = w_tilde / (w_tilde.mean() + 1e-8)
        
        # 5. Clamp weights
        # Prevents any single sample from having too much or too little influence.
        # min=0.1 ensures no sample is completely ignored; max=3.0 prevents gradient explosion.
        # return torch.clamp(w, min=0.1, max=3.0)

        # 5. return weights as it is without clamping
        return w

    # def training_step(self, batch, batch_idx):
    #     x_s, _ = self._split_batch(batch)
    #     x_t, _ = self._split_batch(next(self.target_iter))
    #     x_t = x_t.to(self.device)

    #     if x_s.ndim == 5: x_s = x_s.flatten(1, 2)
    #     if x_t.ndim == 5: x_t = x_t.flatten(1, 2)

    #     # Forward passes
    #     feat_s = self.encoder_s(x_s)[-1] 
    #     feat_t = self.encoder_t(x_t)[-1] 
    #     logits_t = self.seg_head(self.decoder(*self.encoder_t(x_t)))

    #     # Weighted Adversarial Loss (Eq 14 Part 2)
    #     w = self.compute_importance(feat_s)
    #     d0_s = torch.sigmoid(self.domain_adv(self.grl(feat_s)))
    #     d0_t = torch.sigmoid(self.domain_adv(self.grl(feat_t)))
    #     loss_adv = -(w * torch.log(d0_s + 1e-8) + torch.log(1.0 - d0_t + 1e-8)).mean()

    #     # Target Entropy (Eq 13/14 Part 1)
    #     p_t = torch.sigmoid(logits_t)
    #     entropy = -p_t * torch.log(p_t + 1e-8) - (1 - p_t) * torch.log(1 - p_t + 1e-8)
    #     loss_entropy = entropy.mean()

    #     # Total Objective Function: Equation 14
    #     total_loss = (self.hparams.gamma_entropy * loss_entropy) + (self.hparams.lambda_adv * loss_adv)

    #     self.log("train_loss_adv", loss_adv, prog_bar=True, on_epoch=True)
    #     self.log("train_loss_entropy", loss_entropy, prog_bar=True, on_epoch=True)
    #     return total_loss

    # def training_step(self, batch, batch_idx):
    #     # ----------------------------------------------------------
    #     # Load source & target, ensure same batch size
    #     # ----------------------------------------------------------
    #     x_s, y_s = self._split_batch(batch)
    #     x_t, _ = self._split_batch(next(self.target_iter))
    #     x_s, x_t, y_s = x_s.to(self.device), x_t.to(self.device), y_s.to(self.device)

    #     bs = min(x_s.size(0), x_t.size(0))
    #     x_s, y_s, x_t = x_s[:bs], y_s[:bs], x_t[:bs]

    #     # Flatten temporal dim for ResNet encoder if present
    #     if x_s.ndim == 5: x_s = x_s.flatten(1, 2)
    #     if x_t.ndim == 5: x_t = x_t.flatten(1, 2)

    #     # ----------------------------------------------------------
    #     # Forward passes (task + features)
    #     # ----------------------------------------------------------
    #     enc_s = self.encoder_t(x_s)
    #     enc_t = self.encoder_t(x_t)
    #     feat_s, feat_t = enc_s[-1], enc_t[-1]

    #     logits_s = self.seg_head(self.decoder(*enc_s))
    #     logits_t = self.seg_head(self.decoder(*enc_t))

    #     # ----------------------------------------------------------
    #     # Importance weights (IWAN) on source features
    #     # ----------------------------------------------------------
    #     w = self.compute_importance(feat_s).detach()

    #     # ----------------------------------------------------------
    #     # Weighted source segmentation loss (task anchor)
    #     # ----------------------------------------------------------
    #     bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=self.hparams.pos_class_weight)
    #     seg_loss = bce(logits_s.squeeze(1), y_s.float())
    #     # broadcast w over spatial dims
    #     seg_loss = (w.view(-1, *([1] * (seg_loss.ndim - 1))) * seg_loss).mean()

    #     # ----------------------------------------------------------
    #     # Adversarial alignment (weighted on source side)
    #     # ----------------------------------------------------------
    #     d0_s = torch.sigmoid(self.domain_adv(self.grl(feat_s)))
    #     d0_t = torch.sigmoid(self.domain_adv(self.grl(feat_t)))
    #     loss_adv = -(w * torch.log(d0_s + 1e-8) + torch.log(1.0 - d0_t + 1e-8)).mean()

    #     # ----------------------------------------------------------
    #     # Target entropy minimization
    #     # ----------------------------------------------------------
    #     p_t = torch.sigmoid(logits_t)
    #     entropy = -p_t * torch.log(p_t + 1e-8) - (1 - p_t) * torch.log(1 - p_t + 1e-8)
    #     loss_entropy = entropy.mean()

    #     total_loss = seg_loss + self.hparams.lambda_adv * loss_adv + self.hparams.gamma_entropy * loss_entropy

    #     self.log("train_loss_seg", seg_loss, prog_bar=True, on_epoch=True)
    #     self.log("train_loss_adv", loss_adv, prog_bar=True, on_epoch=True)
    #     self.log("train_loss_entropy", loss_entropy, prog_bar=True, on_epoch=True)
    #     return total_loss

    def training_step(self, batch, batch_idx):
        # 1) Load source batch (with labels) and next target batch (unlabeled), keep sizes matched.
        x_s, y_s = self._split_batch(batch)
        x_t, _ = self._split_batch(next(self.target_iter))
        x_s, x_t, y_s = x_s.to(self.device), x_t.to(self.device), y_s.to(self.device)
        bs = min(x_s.size(0), x_t.size(0))
        x_s, y_s, x_t = x_s[:bs], y_s[:bs], x_t[:bs]

        # 2) Flatten temporal dim if present (ResNet expects 4D).
        if x_s.ndim == 5: x_s = x_s.flatten(1, 2)
        if x_t.ndim == 5: x_t = x_t.flatten(1, 2)

        # 3) Forward through the target feature extractor + task head for both domains.
        enc_s = self.encoder_t(x_s)
        enc_t = self.encoder_t(x_t)
        feat_s, feat_t = enc_s[-1], enc_t[-1]
        logits_s = self.seg_head(self.decoder(*enc_s))
        logits_t = self.seg_head(self.decoder(*enc_t))

        # 4) Importance weights (IWAN) on source features to suppress outliers.
        w = self.compute_importance(feat_s).detach()

        # 5) Weighted source segmentation loss (anchors task; head remains trainable).
        pos_w = torch.as_tensor(self.hparams.pos_class_weight, device=self.device)
        bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_w)
        seg_loss = bce(logits_s.squeeze(1), y_s.float())
        seg_loss = (w.view(-1, *([1] * (seg_loss.ndim - 1))) * seg_loss).mean()

        # # Map pos_weight -> alpha in [0,1] for focal; keeps class bias but stable.
        # alpha = float((pos_w / (1.0 + pos_w)).clamp(0.0, 1.0))
        # seg_loss_pix = sigmoid_focal_loss(
        #     logits_s.squeeze(1),
        #     y_s.float(),
        #     alpha=alpha,
        #     gamma=2.0,
        #     reduction="none",
        # )
        # seg_loss = (w.view(-1, *([1] * (seg_loss_pix.ndim - 1))) * seg_loss_pix).mean()

        # 6) Adversarial alignment, weighted on the source side.
        d0_s = torch.sigmoid(self.domain_adv(self.grl(feat_s)))
        d0_t = torch.sigmoid(self.domain_adv(self.grl(feat_t)))
        loss_adv = -(w * torch.log(d0_s + 1e-8) + torch.log(1.0 - d0_t + 1e-8)).mean()

        # 7) Target entropy minimization for confident target predictions.
        p_t = torch.sigmoid(logits_t)
        entropy = -p_t * torch.log(p_t + 1e-8) - (1 - p_t) * torch.log(1 - p_t + 1e-8)
        loss_entropy = entropy.mean()

        lam_adv = self._lambda_adv_factor()
        total_loss = seg_loss + lam_adv * loss_adv + self.hparams.gamma_entropy * loss_entropy

        self.log("train_loss_seg", seg_loss, prog_bar=True, on_epoch=True)
        self.log("train_loss_adv", loss_adv, prog_bar=True, on_epoch=True)
        self.log("train_loss_entropy", loss_entropy, prog_bar=True, on_epoch=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = self._split_batch(batch)
        logits = self.seg_head(self.decoder(*self.encoder_t(x)))
        preds = torch.sigmoid(logits).squeeze(1)
        self.target_iou.update(preds, y.long())
        self.log("val_target_iou", self.target_iou, on_epoch=True, prog_bar=True)

    # def test_step(self, batch, batch_idx):
    #     """ FULL LOGGING: Implementing all metrics from BaseModel """
    #     x, y = self._split_batch(batch)
    #     logits = self.seg_head(self.decoder(*self.encoder_t(x)))
    #     preds = torch.sigmoid(logits).squeeze(1)
    #     y_float = y.float()
    #     y_long = y.long()
        
    #     loss = self.compute_loss(preds, y_float)

    #     # Track every metric as before
    #     self.test_f1(preds, y_long)
    #     self.test_avg_precision(preds, y_long)
    #     self.test_precision(preds, y_long)
    #     self.test_recall(preds, y_long)
    #     self.test_iou(preds, y_long)

    #     self.log("test_loss", loss, prog_bar=True)
    #     self.log_dict({
    #         "test_f1": self.test_f1,
    #         "test_AP": self.test_avg_precision,
    #         "test_precision": self.test_precision,
    #         "test_recall": self.test_recall,
    #         "test_iou": self.test_iou,
    #     })
    #     return loss

    def test_step(self, batch, batch_idx):
        """ FULL LOGGING: Implementing all metrics from BaseModel """
        x, y = self._split_batch(batch)
        
        # FIX: Flatten 5D [B, T, C, H, W] to 4D [B, T*C, H, W] for the ResNet encoder
        if x.ndim == 5: 
            x = x.flatten(1, 2)
            
        # Use adapted Ft and frozen C for final predictions
        logits = self.seg_head(self.decoder(*self.encoder_t(x)))
        preds = torch.sigmoid(logits).squeeze(1)
        
        y_float = y.float()
        y_long = y.long()
        
        loss = self.compute_loss(preds, y_float)

        # Track every metric as before
        self.test_f1(preds, y_long)
        self.test_avg_precision(preds, y_long)
        self.test_precision(preds, y_long)
        self.test_recall(preds, y_long)
        self.test_iou(preds, y_long)

        self.log("test_loss", loss, prog_bar=True)
        self.log_dict({
            "test_f1": self.test_f1,
            "test_AP": self.test_avg_precision,
            "test_precision": self.test_precision,
            "test_recall": self.test_recall,
            "test_iou": self.test_iou,
        })
        return loss

    def configure_optimizers(self):
        # Ft and domain_adv share base LR; decoder/seg_head get a smaller LR for stability.
        head_lr = self.hparams.lr * self.hparams.lr_head_scale
        return torch.optim.Adam([
            {"params": self.encoder_t.parameters(), "lr": self.hparams.lr},
            {"params": self.domain_adv.parameters(), "lr": self.hparams.lr},
            {"params": list(self.decoder.parameters()) + list(self.seg_head.parameters()), "lr": head_lr},
        ])

    def _lambda_adv_factor(self):
        """Linear warmup for lambda_adv to avoid early instability."""
        warmup = max(1, int(self.hparams.lambda_adv_warmup_epochs))
        scale = min(1.0, (self.current_epoch + 1) / warmup)
        return self.hparams.lambda_adv * scale
