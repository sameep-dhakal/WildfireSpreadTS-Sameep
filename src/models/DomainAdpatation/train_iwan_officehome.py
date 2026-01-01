#!/usr/bin/env python3
"""
IWAN (Importance Weighted Adversarial Nets) on Office-Home (Partial Domain Adaptation).

This script is a CLEAN, RUNNABLE reference implementation that:
1) Enforces partial DA: target label space is a strict subset of source (Y_t ⊂ Y_s).
2) Implements two discriminators:
   - D  : for importance weights w(z_s)=1-sigmoid(D(Fs(x_s))) with mean-normalization
   - D0 : for weighted adversarial alignment between Fs(x_s) and Ft(x_t) using GRL + lambda schedule
3) Prevents gradient leakage through weights (stop-grad) and keeps Fs frozen.
4) Trains Ft with:
   - source classification CE loss (anchor)
   - + weighted adversarial loss (D0)
   - + optional target entropy minimization
5) Uses deterministic train/val splits with correct transforms:
   - train: random resized crop + flip
   - val  : center crop

Expected Office-Home layout:
DATA_ROOT/
  Art/ classA/*.jpg ...
  Clipart/
  Product/
  RealWorld/

Example:
python train_iwan_officehome_pda.py \
  --data_root /develop/data/OfficeHomeDataset \
  --source Product --target RealWorld \
  --arch resnet50 --bottleneck_dim 256 \
  --partial_k 25 --partial_seed 0 \
  --epochs_cls 200 --epochs_da 250 \
  --batch_size 64 --lr_cls 0.001 --lr_da 0.001 \
  --lambda_upper 0.1 --alpha 1.0 --entropy_weight 0.0
"""

import argparse
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, Subset


# -------------------------------
# GRL
# -------------------------------
class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd: float):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd: float):
    return GRL.apply(x, lambd)


# -------------------------------
# Models
# -------------------------------
class FeatureBackbone(nn.Module):
    def __init__(self, arch: str = "resnet50"):
        super().__init__()
        if arch == "resnet50":
            net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        elif arch == "resnet18":
            net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported arch: {arch}")
        self.out_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.net = net

    def forward(self, x):
        return self.net(x)


class IWANClassifier(nn.Module):
    """Backbone + optional bottleneck + linear classifier."""
    def __init__(self, num_classes: int, arch: str = "resnet50", bottleneck_dim: int = 256):
        super().__init__()
        self.backbone = FeatureBackbone(arch=arch)
        in_dim = self.backbone.out_dim

        self.use_bottleneck = bottleneck_dim is not None and bottleneck_dim > 0
        if self.use_bottleneck:
            self.bottleneck = nn.Sequential(
                nn.Linear(in_dim, bottleneck_dim),
                nn.ReLU(inplace=True),
            )
            feat_dim = bottleneck_dim
        else:
            self.bottleneck = None
            feat_dim = in_dim

        self.classifier = nn.Linear(feat_dim, num_classes)
        self.feat_dim = feat_dim

    def feats(self, x):
        z = self.backbone(x)
        if self.bottleneck is not None:
            z = self.bottleneck(z)
        return z

    def forward(self, x):
        z = self.feats(x)
        logits = self.classifier(z)
        return logits, z


class DomainDiscriminator(nn.Module):
    """MLP: 1024 -> 1024 -> 1 (common in PDA baselines)."""
    def __init__(self, in_dim: int, hidden: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def freeze_bn(module: nn.Module):
    """Optional stabilization for ResNet BN (not required by paper)."""
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)


# -------------------------------
# Data
# -------------------------------
class OfficeHomeDomain(Dataset):
    """
    ImageFolder wrapper with label IDs forced to SOURCE domain's class_to_idx
    so labels are consistent across domains.
    """
    def __init__(self, domain_root: str, source_class_to_idx: Dict[str, int], transform):
        self.inner = torchvision.datasets.ImageFolder(domain_root, transform=transform)
        self.source_class_to_idx = source_class_to_idx

        # Store remapped samples without loading images
        self.samples: List[Tuple[str, int]] = []
        for path, y_inner in self.inner.samples:
            cls_name = self.inner.classes[y_inner]
            if cls_name in source_class_to_idx:
                self.samples.append((path, source_class_to_idx[cls_name]))

        # use ImageFolder loader
        self.loader = self.inner.loader
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        x = self.loader(path)
        if self.transform is not None:
            x = self.transform(x)
        return x, y


class FilterByClassIDs(Dataset):
    """Filter dataset by label IDs WITHOUT loading images."""
    def __init__(self, base: OfficeHomeDomain, keep_ids: List[int]):
        self.base = base
        keep = set(int(k) for k in keep_ids)
        self.indices = [i for i, (_, y) in enumerate(base.samples) if int(y) in keep]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base[self.indices[idx]]


def make_split_indices(n: int, val_split: float, seed: int = 42) -> Tuple[List[int], List[int]]:
    n_val = int(n * val_split)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    return tr_idx, val_idx


@dataclass
class DataPack:
    src_train: DataLoader
    src_val: DataLoader
    tgt_train: DataLoader
    tgt_val: DataLoader
    num_classes: int
    keep_ids: List[int]
    keep_names: List[str]


def make_loaders(
    data_root: str,
    source: str,
    target: str,
    batch_size: int,
    num_workers: int,
    val_split: float,
    partial_k: int,
    partial_seed: int,
    split_seed: int = 42,
) -> DataPack:
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_tf = T.Compose([
        T.Resize(256),
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])
    val_tf = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize,
    ])

    # Canonical label mapping from source domain classes
    src_probe = torchvision.datasets.ImageFolder(os.path.join(data_root, source), transform=train_tf)
    num_classes = len(src_probe.classes)
    source_class_to_idx = src_probe.class_to_idx
    idx_to_class = {v: k for k, v in source_class_to_idx.items()}

    # Choose target subset of classes (source IDs)
    rng = np.random.default_rng(partial_seed)
    keep_ids = sorted(rng.choice(num_classes, size=partial_k, replace=False).tolist())
    keep_names = [idx_to_class[i] for i in keep_ids]

    # Build datasets with correct transforms (train vs val), same files, split by indices
    src_train_ds = OfficeHomeDomain(os.path.join(data_root, source), source_class_to_idx, transform=train_tf)
    src_val_ds   = OfficeHomeDomain(os.path.join(data_root, source), source_class_to_idx, transform=val_tf)

    tgt_train_ds = OfficeHomeDomain(os.path.join(data_root, target), source_class_to_idx, transform=train_tf)
    tgt_val_ds   = OfficeHomeDomain(os.path.join(data_root, target), source_class_to_idx, transform=val_tf)

    # Filter target to partial subset (PDA condition)
    tgt_train_ds = FilterByClassIDs(tgt_train_ds, keep_ids=keep_ids)
    tgt_val_ds   = FilterByClassIDs(tgt_val_ds, keep_ids=keep_ids)

    # Split indices once per (source, target) to keep deterministic partitioning
    s_tr_idx, s_va_idx = make_split_indices(len(src_train_ds), val_split, seed=split_seed)
    t_tr_idx, t_va_idx = make_split_indices(len(tgt_train_ds), val_split, seed=split_seed)

    src_train = DataLoader(Subset(src_train_ds, s_tr_idx), batch_size=batch_size, shuffle=True,
                           num_workers=num_workers, drop_last=True, pin_memory=True)
    src_val   = DataLoader(Subset(src_val_ds,   s_va_idx), batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, drop_last=False, pin_memory=True)

    tgt_train = DataLoader(Subset(tgt_train_ds, t_tr_idx), batch_size=batch_size, shuffle=True,
                           num_workers=num_workers, drop_last=True, pin_memory=True)
    tgt_val   = DataLoader(Subset(tgt_val_ds,   t_va_idx), batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, drop_last=False, pin_memory=True)

    return DataPack(
        src_train=src_train,
        src_val=src_val,
        tgt_train=tgt_train,
        tgt_val=tgt_val,
        num_classes=num_classes,
        keep_ids=keep_ids,
        keep_names=keep_names,
    )


# -------------------------------
# Metrics
# -------------------------------
@torch.no_grad()
def eval_acc(model: IWANClassifier, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits, _ = model(x)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


@torch.no_grad()
def eval_ce(model: IWANClassifier, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits, _ = model(x)
        loss = ce(logits, y)
        total_loss += loss.item() * y.size(0)
        total += y.size(0)
    return total_loss / max(1, total)


# -------------------------------
# Training
# -------------------------------
def lambda_schedule(progress: float, lambda_upper: float, alpha: float, device: torch.device) -> float:
    # paper-style ramp: 2u/(1+exp(-a p)) - u
    p = torch.tensor(float(progress), device=device)
    val = 2.0 * lambda_upper / (1.0 + torch.exp(-alpha * p)) - lambda_upper
    return float(val.item())


def train_source(
    model: IWANClassifier,
    src_train: DataLoader,
    src_val: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    patience: int,
) -> IWANClassifier:
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * len(src_train)))
    ce = nn.CrossEntropyLoss()

    best = float("inf")
    best_state = None
    bad = 0

    for ep in range(epochs):
        model.train()
        for x, y in src_train:
            x = x.to(device)
            y = y.to(device)
            logits, _ = model(x)
            loss = ce(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

        val_loss = eval_ce(model, src_val, device)
        if val_loss < best:
            best = val_loss
            best_state = deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1

        print(f"[CLS] ep {ep+1}/{epochs} src_val_ce={val_loss:.4f} best={best:.4f}")
        if patience > 0 and bad >= patience:
            print("[CLS] early stop")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_iwan(
    model: IWANClassifier,
    src_train: DataLoader,
    tgt_train: DataLoader,
    src_val: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    lambda_upper: float,
    alpha: float,
    entropy_weight: float,
    patience: int,
    freeze_bn_flag: bool,
):
    """
    IWAN stage:
    - Fs frozen copy of feature extractor (and bottleneck)
    - Ft = feature extractor in model
    - C frozen classifier head
    - D  estimates importance weights on Fs(x_s)
    - D0 adversarial alignment (weighted Fs(x_s) vs Ft(x_t)) with GRL(lambda)
    - Ft trained with: CE(source) + Adv(D0,weighted) + entropy(optional)
    """
    model.to(device)

    # Frozen Fs
    Fs_backbone = deepcopy(model.backbone).to(device).eval()
    for p in Fs_backbone.parameters():
        p.requires_grad_(False)

    Fs_bottleneck = None
    if model.bottleneck is not None:
        Fs_bottleneck = deepcopy(model.bottleneck).to(device).eval()
        for p in Fs_bottleneck.parameters():
            p.requires_grad_(False)

    # Ft and C
    Ft_backbone = model.backbone
    Ft_bottleneck = model.bottleneck
    C = model.classifier
    for p in C.parameters():
        p.requires_grad_(False)

    if freeze_bn_flag:
        freeze_bn(Ft_backbone)

    # Discriminators
    feat_dim = model.feat_dim
    D = DomainDiscriminator(in_dim=feat_dim).to(device)
    D0 = DomainDiscriminator(in_dim=feat_dim).to(device)

    opt_D = torch.optim.SGD(D.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    opt_main = torch.optim.SGD(
        list(Ft_backbone.parameters())
        + ([] if Ft_bottleneck is None else list(Ft_bottleneck.parameters()))
        + list(D0.parameters()),
        lr=lr, momentum=0.9, weight_decay=5e-4
    )

    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss(reduction="none")

    steps_per_epoch = min(len(src_train), len(tgt_train))
    global_step = 0

    best = float("inf")
    best_state = None
    bad = 0

    for ep in range(epochs):
        model.train()
        D.train()
        D0.train()

        it_s = iter(src_train)
        it_t = iter(tgt_train)

        for _ in range(steps_per_epoch):
            x_s, y_s = next(it_s)
            x_t, _ = next(it_t)
            x_s = x_s.to(device)
            y_s = y_s.to(device)
            x_t = x_t.to(device)

            # 1) Fs forward (no grad)
            with torch.no_grad():
                z_s_fs = Fs_backbone(x_s)
                if Fs_bottleneck is not None:
                    z_s_fs = Fs_bottleneck(z_s_fs)

            # 2) Ft forward
            z_t_ft = Ft_backbone(x_t)
            z_s_ft = Ft_backbone(x_s)
            if Ft_bottleneck is not None:
                z_t_ft = Ft_bottleneck(z_t_ft)
                z_s_ft = Ft_bottleneck(z_s_ft)

            logits_s = C(z_s_ft)
            logits_t = C(z_t_ft)

            # 3) Train D: Fs(x_s)=1, Ft(x_t)=0
            log_s_D = D(z_s_fs.detach())
            log_t_D = D(z_t_ft.detach())
            ones = torch.ones_like(log_s_D)
            zeros = torch.zeros_like(log_t_D)

            loss_D = 0.5 * (bce(log_s_D, ones).mean() + bce(log_t_D, zeros).mean())
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # 4) weights from D(Fs(x_s)) (stop-grad), normalize mean to 1
            with torch.no_grad():
                w = 1.0 - torch.sigmoid(log_s_D)
                w = w / (w.mean() + 1e-8)

            # 5) lambda schedule + D0 adversarial
            progress = min(1.0, global_step / max(1, epochs * steps_per_epoch))
            lam = lambda_schedule(progress, lambda_upper=lambda_upper, alpha=alpha, device=device)

            log_s_D0 = D0(z_s_fs.detach())
            log_t_D0 = D0(grad_reverse(z_t_ft, lam))

            loss_s_D0 = (w * bce(log_s_D0, ones)).mean()
            loss_t_D0 = bce(log_t_D0, zeros).mean()
            loss_adv = 0.5 * (loss_s_D0 + loss_t_D0)

            # 6) source CE anchor
            loss_cls = ce(logits_s, y_s)

            # 7) optional target entropy
            loss_ent = torch.tensor(0.0, device=device)
            if entropy_weight > 0:
                p_t = F.softmax(logits_t, dim=1)
                loss_ent = -(p_t * torch.log(p_t + 1e-8)).sum(dim=1).mean()

            # 8) main update
            loss_main = loss_cls + loss_adv + entropy_weight * loss_ent
            opt_main.zero_grad()
            loss_main.backward()
            opt_main.step()

            global_step += 1

        # monitor using source val CE
        src_ce = eval_ce(model, src_val, device)
        print(
            f"[DA] ep {ep+1}/{epochs} src_val_ce={src_ce:.4f} "
            f"D={loss_D.item():.4f} adv={loss_adv.item():.4f} lam={lam:.4f} "
            f"w_mean={w.mean().item():.4f} w_std={w.std().item():.4f} "
            f"w_min={w.min().item():.4f} w_max={w.max().item():.4f}"
        )

        if src_ce < best:
            best = src_ce
            best_state = deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
        if patience > 0 and bad >= patience:
            print("[DA] early stop")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# -------------------------------
# CLI
# -------------------------------
def parse_args():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--source", type=str, required=True, choices=["Art", "Clipart", "Product", "RealWorld"])
    p.add_argument("--target", type=str, required=True, choices=["Art", "Clipart", "Product", "RealWorld"])
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_split", type=float, default=0.1)

    # partial DA
    p.add_argument("--partial_k", type=int, default=25, help="Target uses only K classes (subset of source).")
    p.add_argument("--partial_seed", type=int, default=0, help="Seed for choosing the K target classes.")
    p.add_argument("--split_seed", type=int, default=42, help="Seed for train/val splitting.")

    # model
    p.add_argument("--arch", type=str, default="resnet50", choices=["resnet18", "resnet50"])
    p.add_argument("--bottleneck_dim", type=int, default=256)
    p.add_argument("--freeze_bn", type=int, default=1, help="1 to freeze BatchNorm stats during DA stage (ResNet).")

    # stage 1
    p.add_argument("--epochs_cls", type=int, default=200)
    p.add_argument("--lr_cls", type=float, default=1e-3)
    p.add_argument("--patience_cls", type=int, default=20)

    # stage 2
    p.add_argument("--epochs_da", type=int, default=250)
    p.add_argument("--lr_da", type=float, default=1e-3)
    p.add_argument("--patience_da", type=int, default=0)
    p.add_argument("--lambda_upper", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--entropy_weight", type=float, default=0.0)

    return p.parse_args()


def main():
    args = parse_args()
    if args.source == args.target:
        raise ValueError("source and target must be different domains.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    data = make_loaders(
        data_root=args.data_root,
        source=args.source,
        target=args.target,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        partial_k=args.partial_k,
        partial_seed=args.partial_seed,
        split_seed=args.split_seed,
    )

    print("\n== Partial DA setup ==")
    print(f"Source total classes: {data.num_classes}")
    print(f"Target subset K: {len(data.keep_ids)}")
    print("Target subset class names (first 20):", data.keep_names[:20])

    model = IWANClassifier(
        num_classes=data.num_classes,
        arch=args.arch,
        bottleneck_dim=args.bottleneck_dim,
    )

    # Stage 1: source pretrain
    print("\n==> Stage 1: Source pretrain")
    model = train_source(
        model=model,
        src_train=data.src_train,
        src_val=data.src_val,
        device=device,
        epochs=args.epochs_cls,
        lr=args.lr_cls,
        patience=args.patience_cls,
    )
    src_acc = eval_acc(model, data.src_val, device)
    tgt_acc = eval_acc(model, data.tgt_val, device)  # target already filtered to shared classes only
    print(f"[After CLS] src_val_acc={src_acc:.4f} tgt_val_acc(shared only)={tgt_acc:.4f}")

    # Stage 2: IWAN
    print("\n==> Stage 2: IWAN adaptation")
    model = train_iwan(
        model=model,
        src_train=data.src_train,
        tgt_train=data.tgt_train,
        src_val=data.src_val,
        device=device,
        epochs=args.epochs_da,
        lr=args.lr_da,
        lambda_upper=args.lambda_upper,
        alpha=args.alpha,
        entropy_weight=args.entropy_weight,
        patience=args.patience_da,
        freeze_bn_flag=bool(args.freeze_bn),
    )

    tgt_acc_post = eval_acc(model, data.tgt_val, device)
    print(f"[After IWAN] tgt_val_acc(shared only)={tgt_acc_post:.4f}")


if __name__ == "__main__":
    main()
