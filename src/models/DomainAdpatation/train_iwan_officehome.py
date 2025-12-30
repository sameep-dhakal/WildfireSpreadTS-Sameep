"""
IWAN domain adaptation on Office-Home classification with ResNet-18 backbone.
- Stage 1: train classifier on source domain, save best.
- Stage 2: IWAN adaptation with frozen Fs (source encoder) + trainable Ft, D, D0.
  Uses importance weights w = 1 - sigmoid(D(Fs(x_s))) (normalized), GRL on target,
  and source classification loss as anchor. Lambda for D0 follows paper schedule.

Dataset layout expected (Office-Home style):
    DATA_ROOT/
        Art/
            class1/img1.jpg ...
        Clipart/
        Product/
        RealWorld/

Run example:
    python scripts/train_iwan_officehome.py --data_root /path/OfficeHome \
        --source Art --target Clipart --batch_size 32 --epochs_cls 10 --epochs_da 20

No automatic download (network-restricted); place data manually.
"""
import argparse
import os
from copy import deepcopy
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import wandb
import yaml


# -------------------------------
# Models
# -------------------------------
class ClassifierNet(nn.Module):
    def __init__(self, num_classes: int, arch: str = "resnet18", bottleneck_dim: int = 0):
        super().__init__()
        if arch == "resnet50":
            backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        in_feat = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        if bottleneck_dim and bottleneck_dim > 0:
            self.bottleneck = nn.Sequential(
                nn.Linear(in_feat, bottleneck_dim),
                nn.ReLU(inplace=True),
            )
            head_in = bottleneck_dim
        else:
            self.bottleneck = None
            head_in = in_feat
        self.classifier = nn.Linear(head_in, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        if self.bottleneck is not None:
            feats = self.bottleneck(feats)
        logits = self.classifier(feats)
        return logits, feats


class DomainDiscriminator(nn.Module):
    """Two-layer MLP as in IWAN (default) or deeper 3x1024 if head='mlp3'."""
    def __init__(self, in_dim: int = 512, hidden: int = 1024, head: str = "iwan"):
        super().__init__()
        if head.lower() == "mlp3":
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(hidden, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(hidden, 1),
            )
        else:  # default IWAN 2-layer
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, 1),
            )

    def forward(self, x):
        return self.net(x).squeeze(1)


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GRL.apply(x, lambd)


# -------------------------------
# Data
# -------------------------------
def make_loaders(root: str, source: str, target: str, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader, int]:
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = T.Compose([
        T.Resize(256),
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])
    test_tf = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize,
    ])

    src_ds = torchvision.datasets.ImageFolder(os.path.join(root, source), transform=train_tf)
    tgt_ds = torchvision.datasets.ImageFolder(os.path.join(root, target), transform=train_tf)

    src_loader = DataLoader(src_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    tgt_loader = DataLoader(tgt_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    num_classes = len(src_ds.classes)
    return src_loader, tgt_loader, num_classes


# -------------------------------
# Training loops
# -------------------------------
def train_cls(model, loader, device, epochs, lr, patience=0):
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader))
    ce = nn.CrossEntropyLoss()
    best_acc = 0.0
    best_state = None
    no_improve = 0

    for ep in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = ce(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        acc = eval_acc(model, loader, device)
        if acc > best_acc:
            best_acc = acc
            best_state = deepcopy(model.state_dict())
            no_improve = 0
        print(f"[CLS] Epoch {ep+1}/{epochs} acc={acc:.4f} best={best_acc:.4f}")
        no_improve += 1
        if patience > 0 and no_improve >= patience:
            print(f"[CLS] Early stopping at epoch {ep+1} (patience={patience})")
            break
    if best_state:
        model.load_state_dict(best_state)
    return model


def eval_acc(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(1, total)


def train_iwan(model, loader_s, loader_t, device, epochs, lr, lambda_upper=0.1, alpha=1.0, entropy_weight=0.0, domain_head: str = "iwan", patience=0):
    # Freeze Fs (teacher) and classifier C; only Ft, D, D0 update (paper setup)
    Fs = deepcopy(model.backbone).to(device).eval()
    for p in Fs.parameters():
        p.requires_grad_(False)
    Fs_bottleneck = None
    if hasattr(model, "bottleneck") and model.bottleneck is not None:
        Fs_bottleneck = deepcopy(model.bottleneck).to(device).eval()
        for p in Fs_bottleneck.parameters():
            p.requires_grad_(False)

    Ft = model.backbone
    bottleneck = model.bottleneck
    C = model.classifier
    for p in C.parameters():
        p.requires_grad_(False)

    D = DomainDiscriminator(in_dim=C.in_features, head=domain_head).to(device)
    D0 = DomainDiscriminator(in_dim=C.in_features, head=domain_head).to(device)

    main_params = list(Ft.parameters()) + list(D0.parameters())
    if bottleneck is not None:
        main_params += list(bottleneck.parameters())
    opt_D = torch.optim.SGD(D.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    opt_main = torch.optim.SGD(main_params, lr=lr, momentum=0.9, weight_decay=5e-4)

    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss(reduction="none")

    steps_per_epoch = min(len(loader_s), len(loader_t))
    global_step = 0
    best_tgt_acc = 0.0
    no_improve = 0
    for ep in range(epochs):
        it_s = iter(loader_s)
        it_t = iter(loader_t)
        for _ in range(steps_per_epoch):
            x_s, y_s = next(it_s)
            x_t, _ = next(it_t)
            x_s, y_s, x_t = x_s.to(device), y_s.to(device), x_t.to(device)

            # Forward Fs for weights
            with torch.no_grad():
                zs = Fs(x_s)
                if Fs_bottleneck is not None:
                    zs = Fs_bottleneck(zs)
            # Forward Ft (with bottleneck if present)
            ft_s = Ft(x_s)
            ft_t = Ft(x_t)
            if bottleneck is not None:
                ft_s = bottleneck(ft_s)
                ft_t = bottleneck(ft_t)
            logits_s = C(ft_s)
            logits_t = C(ft_t)

            # D step
            log_s_D = D(zs.detach())
            log_t_D = D(ft_t.detach())
            ones_s = torch.ones_like(log_s_D)
            zeros_t = torch.zeros_like(log_t_D)
            loss_D = 0.5 * (bce(log_s_D, ones_s).mean() + bce(log_t_D, zeros_t).mean())
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # Weights
            with torch.no_grad():
                w_s = 1.0 - torch.sigmoid(log_s_D)
                w_s = w_s / (w_s.mean() + 1e-8)

            # D0 loss with GRL
            log_s_D0 = D0(zs.detach())
            log_t_D0 = D0(grad_reverse(ft_t, lambd=1.0))
            loss_s_D0 = (w_s * bce(log_s_D0, ones_s)).mean()
            loss_t_D0 = bce(log_t_D0, zeros_t).mean()
            loss_D0 = 0.5 * (loss_s_D0 + loss_t_D0)

            # No CE anchor in IWAN stage (C frozen)
            seg_loss = torch.tensor(0.0, device=device)

            # Entropy on target (optional, Eq. 13)
            ent_loss = 0.0
            if entropy_weight > 0:
                p_t = F.softmax(logits_t, dim=1)
                ent = - (p_t * torch.log(p_t + 1e-8)).sum(dim=1).mean()
                ent_loss = ent

            # Lambda schedule
            progress = min(1.0, global_step / max(1, epochs * steps_per_epoch))
            lambda_sched = 2 * lambda_upper / (1.0 + torch.exp(torch.tensor(-alpha * progress, device=device))) - lambda_upper

            loss_main = seg_loss + lambda_sched * loss_D0 + entropy_weight * ent_loss

            opt_main.zero_grad()
            loss_main.backward()
            opt_main.step()

            global_step += 1
        acc = eval_acc(model, loader_t, device)
        # Lightweight stats
        w_mean = w_s.mean().item()
        w_std = w_s.std().item()
        w_min = w_s.min().item()
        w_max = w_s.max().item()
        print(f"[DA] Epoch {ep+1}/{epochs} target_acc={acc:.4f} D={loss_D.item():.4f} D0={loss_D0.item():.4f} lam={lambda_sched.item():.4f} w_mean={w_mean:.4f} w_std={w_std:.4f} w_min={w_min:.4f} w_max={w_max:.4f}")
        if acc > best_tgt_acc:
            best_tgt_acc = acc
            no_improve = 0
        else:
            no_improve += 1
        if patience > 0 and no_improve >= patience:
            print(f"[DA] Early stopping at epoch {ep+1} (patience={patience})")
            break
    return model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="Optional YAML config path.")
    p.add_argument("--data_root", type=str, default=None, help="Path to Office-Home root.")
    p.add_argument("--source", type=str, default=None, help="Source domain (e.g., Art)")
    p.add_argument("--target", type=str, default=None, help="Target domain (e.g., Clipart)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs_cls", type=int, default=50)
    p.add_argument("--epochs_da", type=int, default=80)
    p.add_argument("--patience_cls", type=int, default=0, help="Early stop patience for source pretrain (0=off).")
    p.add_argument("--patience_da", type=int, default=0, help="Early stop patience for DA on target acc (0=off).")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--arch", type=str, default="resnet50", choices=["resnet18", "resnet50"])
    p.add_argument("--bottleneck_dim", type=int, default=256)
    p.add_argument("--lambda_upper", type=float, default=0.1)
    p.add_argument("--entropy_weight", type=float, default=0.0)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--domain_head", type=str, default="iwan", choices=["iwan", "mlp3"], help="Discriminator head type.")
    p.add_argument("--wandb_project", type=str, default=None, help="If set, log metrics to this W&B project.")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--no_wandb", action="store_true", help="Disable wandb logging.")
    defaults = p.parse_args([])
    args = p.parse_args()
    # Merge YAML config if provided: YAML fills in values where CLI left defaults
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
        for k, v in cfg.items():
            if hasattr(args, k):
                if getattr(args, k) == getattr(defaults, k):
                    setattr(args, k, v)
    # Validate required fields
    missing = [k for k in ["data_root", "source", "target"] if getattr(args, k) is None]
    if missing and os.getenv("WANDB_SWEEP_ID") is None:
        raise ValueError(f"Missing required arguments: {missing}. Pass via CLI, config, or sweep.")
    return args


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_wandb = (not args.no_wandb) and (args.wandb_project or os.getenv("WANDB_PROJECT") or os.getenv("WANDB_SWEEP_ID"))
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            config=vars(args),
        )

    src_loader, tgt_loader, num_classes = make_loaders(
        args.data_root, args.source, args.target, args.batch_size, args.num_workers
    )

    model = ClassifierNet(num_classes=num_classes, arch=args.arch, bottleneck_dim=args.bottleneck_dim)

    # Stage-1: source pretrain
    print("==> Stage-1: source pretraining")
    model = train_cls(model, src_loader, device, epochs=args.epochs_cls, lr=args.lr, patience=args.patience_cls)
    src_acc = eval_acc(model, src_loader, device)
    tgt_acc = eval_acc(model, tgt_loader, device)
    print(f"Source pretrain done. Src acc={src_acc:.4f} Tgt acc={tgt_acc:.4f}")
    if use_wandb:
        wandb.log({"src_acc_pretrain": src_acc, "tgt_acc_pretrain": tgt_acc})

    # Stage-2: IWAN
    print("==> Stage-2: IWAN adaptation")
    model = train_iwan(
        model,
        loader_s=src_loader,
        loader_t=tgt_loader,
        device=device,
        epochs=args.epochs_da,
        lr=args.lr,
        lambda_upper=args.lambda_upper,
        alpha=args.alpha,
        entropy_weight=args.entropy_weight,
        domain_head=args.domain_head,
        patience=args.patience_da,
    )
    tgt_acc = eval_acc(model, tgt_loader, device)
    print(f"Post-adaptation target acc={tgt_acc:.4f}")
    if use_wandb:
        wandb.log({"tgt_acc_post_da": tgt_acc})
        wandb.finish()


if __name__ == "__main__":
    main()
