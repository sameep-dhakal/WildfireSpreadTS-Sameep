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
        gamma_entropy: float = 0.1,  # γ in Eq 14
        lambda_adv: float = 1.0,     # λ in Eq 14
        lr: float = 1e-4,
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
        for p in self.decoder.parameters(): p.requires_grad = False
        for p in self.seg_head.parameters(): p.requires_grad = False

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

    @torch.no_grad()
    def compute_importance(self, feat):
        """ Eq 7 & 8: Importance weights w(z) """
        d_out = torch.sigmoid(self.domain_oracle(feat))
        w_tilde = (1.0 - d_out) / (d_out + 1e-8)
        # Normalization as per Eq 8
        return w_tilde / (w_tilde.mean() + 1e-8)

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

    def training_step(self, batch, batch_idx):
        # A. Load Source & Target Streams
        x_s, _ = self._split_batch(batch)
        target_batch = next(self.target_iter)
        x_t, _ = self._split_batch(target_batch)
        x_t = x_t.to(self.device)

        # --- FIX: ENSURE BATCH SIZES MATCH ---
        # Find the smaller of the two batch sizes (e.g., 58 vs 64)
        batch_size = min(x_s.size(0), x_t.size(0))
        
        # Truncate both to the same size so the math in Eq 14 works
        x_s = x_s[:batch_size]
        x_t = x_t[:batch_size]
        # -------------------------------------

        if x_s.ndim == 5: x_s = x_s.flatten(1, 2)
        if x_t.ndim == 5: x_t = x_t.flatten(1, 2)

        # B. Forward Passes [cite: 91, 140]
        feat_s = self.encoder_s(x_s)[-1] 
        feat_t = self.encoder_t(x_t)[-1] 
        logits_t = self.seg_head(self.decoder(*self.encoder_t(x_t)))

        # C. Weighted Adversarial Loss (Eq 14 Part 2) [cite: 163]
        w = self.compute_importance(feat_s)
        
        # Now w (size 58), d0_s (size 58), and d0_t (size 58) will match perfectly!
        d0_s = torch.sigmoid(self.domain_adv(self.grl(feat_s)))
        d0_t = torch.sigmoid(self.domain_adv(self.grl(feat_t)))
        
        # This line will no longer throw an error
        loss_adv = -(w * torch.log(d0_s + 1e-8) + torch.log(1.0 - d0_t + 1e-8)).mean()

        # D. Target Entropy (Eq 13/14 Part 1) [cite: 154, 163]
        p_t = torch.sigmoid(logits_t)
        entropy = -p_t * torch.log(p_t + 1e-8) - (1 - p_t) * torch.log(1 - p_t + 1e-8)
        loss_entropy = entropy.mean()

        # E. Total Objective Function: Equation 14 [cite: 164]
        total_loss = (self.hparams.gamma_entropy * loss_entropy) + (self.hparams.lambda_adv * loss_adv)

        self.log("train_loss_adv", loss_adv, prog_bar=True, on_epoch=True)
        self.log("train_loss_entropy", loss_entropy, prog_bar=True, on_epoch=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = self._split_batch(batch)
        logits = self.seg_head(self.decoder(*self.encoder_t(x)))
        preds = torch.sigmoid(logits).squeeze(1)
        self.target_iou.update(preds, y.long())
        self.log("val_target_iou", self.target_iou, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """ FULL LOGGING: Implementing all metrics from BaseModel """
        x, y = self._split_batch(batch)
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
        # Optimized in stages: ONLY Ft and D0 as per paper Section 3.2
        return torch.optim.Adam([
            {'params': self.encoder_t.parameters(), 'lr': self.hparams.lr},
            {'params': self.domain_adv.parameters(), 'lr': self.hparams.lr},
        ])