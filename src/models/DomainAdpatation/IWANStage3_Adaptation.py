import os, copy, torch
import torch.nn as nn, torch.nn.functional as F
import segmentation_models_pytorch as smp
from ..BaseModel import BaseModel


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd): ctx.lambd = lambd; return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return -ctx.lambd * g, None
class GradRev(nn.Module):
    def __init__(self, lambd=1.0): super().__init__(); self.lambd=lambd
    def forward(self, x): return GRL.apply(x, self.lambd)


class DomainLogitHead(nn.Module):
    def __init__(self, in_ch=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(in_ch, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 1),
        )
    def forward(self, x): return self.net(x).squeeze(1)


class IWANStage3_Adaptation(BaseModel):
    """IWAN Stage 3 – Adversarial adaptation (Ft + D₀) with GRL and entropy."""
    def __init__(self, ckpt_dir_stage1, ckpt_dir_stage2, encoder_name="resnet18",
                 n_channels=7, lambda_adv=1.0, gamma_entropy=0.01, **kwargs):
        super().__init__(n_channels=n_channels, **kwargs)
        self.save_hyperparameters()

        # ---- load Fs + C ----
        f1 = [f for f in os.listdir(ckpt_dir_stage1) if f.endswith(".ckpt")]
        if not f1: raise FileNotFoundError(f"No .ckpt in {ckpt_dir_stage1}")
        ckpt_fs = torch.load(os.path.join(ckpt_dir_stage1, f1[0]), map_location="cpu")
        src = smp.Unet(encoder_name, weights=None, in_channels=n_channels, classes=1)
        sd = ckpt_fs.get("state_dict", ckpt_fs)
        src.load_state_dict({k.replace("model.", ""): v for k, v in sd.items() if "model." in k}, strict=False)
        self.Fs = src.encoder.eval()
        self.decoder, self.seg_head = src.decoder, src.segmentation_head
        for p in self.Fs.parameters(): p.requires_grad_(False)

        # ---- load D_weights ----
        f2 = [f for f in os.listdir(ckpt_dir_stage2) if f.endswith(".ckpt")]
        if not f2: raise FileNotFoundError(f"No .ckpt in {ckpt_dir_stage2}")
        ckpt_d = torch.load(os.path.join(ckpt_dir_stage2, f2[0]), map_location="cpu")
        self.Dw = DomainLogitHead(self.Fs.out_channels[-1])
        self.Dw.load_state_dict({k.replace("D.", ""): v for k, v in ckpt_d["state_dict"].items() if "D." in k})
        self.Dw.eval()
        for p in self.Dw.parameters(): p.requires_grad_(False)

        # ---- initialize Ft + D₀ ----
        self.Ft = copy.deepcopy(self.Fs)
        self.D0 = DomainLogitHead(self.Fs.out_channels[-1])
        self.grl = GradRev(1.0)
        self.bce = nn.BCEWithLogitsLoss()
        self.lambda_adv, self.gamma_entropy = lambda_adv, gamma_entropy
        print("✅ Stage 3 initialized (Fs,C,Dw loaded)")

    def _entropy(self, logits):
        p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
        return -(p * torch.log(p) + (1 - p) * torch.log(1 - p)).mean()

    def on_train_start(self):
        self.target_iter = iter(self.trainer.datamodule.target_dataloader())

    def training_step(self, batch, _):
        x_s, _ = batch[0], batch[1]
        try: x_t, _ = next(self.target_iter)
        except StopIteration:
            self.target_iter = iter(self.trainer.datamodule.target_dataloader())
            x_t, _ = next(self.target_iter)
        x_t = x_t.to(self.device, non_blocking=True)

        with torch.no_grad(): f_s = self.Fs(x_s)[-1]
        f_t = self.Ft(x_t)[-1]

        with torch.no_grad():
            p_src = torch.sigmoid(self.Dw(f_s))
            w = (1 - p_src) / ((1 - p_src).mean() + 1e-6)

        log_s0 = self.D0(f_s)
        log_t0 = self.D0(self.grl(f_t))
        loss_src = F.binary_cross_entropy_with_logits(log_s0, torch.ones_like(log_s0), weight=w)
        loss_tgt = F.binary_cross_entropy_with_logits(log_t0, torch.zeros_like(log_t0))
        loss_adv = loss_src + loss_tgt

        dec = self.decoder(*self.Ft(x_t))
        y_hat_t = self.seg_head(dec)
        loss_ent = self._entropy(y_hat_t)

        total = self.lambda_adv * loss_adv + self.gamma_entropy * loss_ent
        self.log_dict({"loss_adv": loss_adv, "loss_ent": loss_ent, "loss_total": total})
        return total

    def configure_optimizers(self):
        return torch.optim.Adam(list(self.Ft.parameters()) + list(self.D0.parameters()), lr=1e-4)
