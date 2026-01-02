#!/usr/bin/env python3
"""
IWAN on Office-31 (Office-Caltech-10 shared classes) with AlexNet+bottleneck.

This script follows the *standard DA protocol (Option A)* used for tables like:
A31→W10, D31→W10, W31→D10, A31→D10, D31→A10, W31→A10

Protocol:
- Source train (labeled): ALL source images, ALL 31 classes
- Target train (unlabeled): ALL target images, BUT FILTERED to the fixed 10 shared classes
- Target test (evaluation only): ALL target images, same 10 shared classes (labels used only for reporting)

You can reproduce:
- "AlexNet+bottleneck" = Stage 1 only (no adaptation)
- "proposed (γ=0)"      = IWAN with entropy_weight=0.0
- "proposed"            = IWAN with entropy_weight>0 (target entropy minimization)

Dataset layout expected:
DATA_ROOT/
  amazon/<class_name>/*.jpg
  dslr/<class_name>/*.jpg
  webcam/<class_name>/*.jpg
Class folder names should match the Office-31 naming (underscore style).

Example:
python3 train_iwan_office31.py \
  --data_root /develop/data/Office-31 \
  --source amazon --target webcam \
  --arch alexnet --bottleneck_dim 256 \
  --epochs_cls 200 --epochs_da 250 \
  --lr_cls 0.001 --lr_da 0.001 \
  --lambda_upper 0.1 --alpha 1.0 \
  --entropy_weight 0.0
"""

import argparse
import os
from copy import deepcopy
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset


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
# AlexNet backbone (feature extractor)
# -------------------------------
class AlexNetFeat(nn.Module):
    """
    Returns a 4096-d feature (fc7 output) from ImageNet pretrained AlexNet.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = torchvision.models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.alexnet(weights=weights)
        self.features = net.features
        self.avgpool = net.avgpool
        # classifier = [Dropout, Linear(9216->4096), ReLU, Dropout, Linear(4096->4096), ReLU, Linear(4096->1000)]
        self.classifier_prefix = nn.Sequential(*list(net.classifier.children())[:6])  # up to fc7 ReLU
        self.out_dim = 4096

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier_prefix(x)
        return x


class IWANClassifier(nn.Module):
    """
    AlexNet feature -> bottleneck -> classifier (31-way).
    """
    def __init__(self, num_classes: int, bottleneck_dim: int = 256, pretrained: bool = True):
        super().__init__()
        self.backbone = AlexNetFeat(pretrained=pretrained)
        in_dim = self.backbone.out_dim

        self.use_bottleneck = bottleneck_dim is not None and bottleneck_dim > 0
        if self.use_bottleneck:
            self.bottleneck = nn.Sequential(
                nn.Linear(in_dim, bottleneck_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
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
    """
    MLP discriminator: in -> 1024 -> 1024 -> 1
    """
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


# -------------------------------
# Data: Office31 with canonical label mapping from source domain
# -------------------------------
class Office31Domain(Dataset):
    """
    ImageFolder wrapper with label IDs forced to SOURCE domain's class_to_idx
    so labels are consistent across domains.
    """
    def __init__(self, domain_root: str, source_class_to_idx: Dict[str, int], transform):
        self.inner = torchvision.datasets.ImageFolder(domain_root, transform=transform)
        self.source_class_to_idx = source_class_to_idx

        self.samples: List[Tuple[str, int]] = []
        for path, y_inner in self.inner.samples:
            cls_name = self.inner.classes[y_inner]
            if cls_name in source_class_to_idx:
                self.samples.append((path, source_class_to_idx[cls_name]))

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
    def __init__(self, base: Office31Domain, keep_ids: List[int]):
        self.base = base
        keep = set(int(k) for k in keep_ids)
        self.indices = [i for i, (_, y) in enumerate(base.samples) if int(y) in keep]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base[self.indices[idx]]


def get_shared10_list(mode: str) -> List[str]:
    """
    Fixed Office-Caltech-10 shared classes.
    These names match the common folder naming used in Office-31 / Office-Caltech splits.
    """
    if mode != "office_caltech_10":
        raise ValueError(f"Unknown partial_mode: {mode}")

    return [
        "back_pack",
        "bike",
        "calculator",
        "headphones",
        "keyboard",
        "laptop_computer",
        "monitor",
        "mouse",
        "mug",
        "projector",
    ]


def make_loaders_optionA(
    data_root: str,
    source: str,
    target: str,
    batch_size: int,
    num_workers: int,
    partial_mode: str,
) -> Tuple[DataLoader, DataLoader, DataLoader, int, List[str]]:
    """
    Option A:
    - src_train = all source (31 classes)
    - tgt_train = all target filtered to shared-10 (unlabeled in training)
    - tgt_test  = all target filtered to shared-10 (for evaluation only)
    """
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

    src_root = os.path.join(data_root, source)
    tgt_root = os.path.join(data_root, target)

    if not os.path.isdir(src_root):
        raise FileNotFoundError(f"Missing source domain folder: {src_root}")
    if not os.path.isdir(tgt_root):
        raise FileNotFoundError(f"Missing target domain folder: {tgt_root}")

    # Canonical mapping from source domain folder names
    src_probe = torchvision.datasets.ImageFolder(src_root, transform=train_tf)
    source_class_to_idx = src_probe.class_to_idx
    num_classes = len(src_probe.classes)

    # Full source
    src_ds = Office31Domain(src_root, source_class_to_idx, transform=train_tf)

    # Target filtered to shared-10
    shared_names = get_shared10_list(partial_mode)
    missing = [c for c in shared_names if c not in source_class_to_idx]
    if missing:
        raise ValueError(
            "Your source domain does not contain some shared-10 classes. "
            f"Missing in source mapping: {missing}\n"
            "This usually means your folder names differ from expected."
        )

    keep_ids = [source_class_to_idx[c] for c in shared_names]

    tgt_train_full = Office31Domain(tgt_root, source_class_to_idx, transform=train_tf)
    tgt_test_full  = Office31Domain(tgt_root, source_class_to_idx, transform=test_tf)

    tgt_train_ds = FilterByClassIDs(tgt_train_full, keep_ids)
    tgt_test_ds  = FilterByClassIDs(tgt_test_full, keep_ids)

    src_train = DataLoader(
        src_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True, pin_memory=True
    )
    tgt_train = DataLoader(
        tgt_train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True, pin_memory=True
    )
    tgt_test = DataLoader(
        tgt_test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False, pin_memory=True
    )

    return src_train, tgt_train, tgt_test, num_classes, shared_names


# -------------------------------
# Metrics
# -------------------------------
@torch.no_grad()
def eval_acc(model: IWANClassifier, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
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
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits, _ = model(x)
        loss = ce(logits, y)
        total_loss += loss.item() * y.size(0)
        total += y.size(0)
    return total_loss / max(1, total)


# -------------------------------
# Training helpers
# -------------------------------
def lambda_schedule(progress: float, lambda_upper: float, alpha: float, device: torch.device) -> float:
    # DANN-style ramp: u * (2/(1+exp(-a p)) - 1)
    p = torch.tensor(float(progress), device=device)
    val = lambda_upper * (2.0 / (1.0 + torch.exp(-alpha * p)) - 1.0)
    return float(val.item())


def train_source(
    model: IWANClassifier,
    src_train: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> IWANClassifier:
    model.to(device)
    model.train()

    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    ce = nn.CrossEntropyLoss()

    for ep in range(epochs):
        model.train()
        running = 0.0
        n = 0
        for x, y in src_train:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits, _ = model(x)
            loss = ce(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += loss.item() * y.size(0)
            n += y.size(0)

        print(f"[CLS] ep {ep+1:03d}/{epochs} src_train_ce={running/max(1,n):.4f}")

    return model


def train_iwan(
    model: IWANClassifier,
    src_train: DataLoader,
    tgt_train: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    lambda_upper: float,
    alpha: float,
    entropy_weight: float,
):
    """
    IWAN:
    - Fs frozen copy of feature extractor (+ bottleneck)
    - Ft = model backbone (+ bottleneck)
    - C frozen classifier
    - D  trains to discriminate Fs(xs) vs Ft(xt) and produces weights: w = 1 - sigmoid(D(Fs(xs)))
    - D0 adversarial alignment between Fs(xs) and Ft(xt) with GRL + weighting on source
    - Ft trained with: CE(source) + Adv(D0) + entropy_weight * Entropy(target)
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
    total_steps = max(1, epochs * steps_per_epoch)

    for ep in range(epochs):
        model.train()
        D.train()
        D0.train()

        it_s = iter(src_train)
        it_t = iter(tgt_train)

        # just for logging the last batch values
        last = {}

        for _ in range(steps_per_epoch):
            x_s, y_s = next(it_s)
            x_t, _ = next(it_t)
            x_s = x_s.to(device, non_blocking=True)
            y_s = y_s.to(device, non_blocking=True)
            x_t = x_t.to(device, non_blocking=True)

            # ---- Fs forward (no grad) on source
            with torch.no_grad():
                z_s_fs = Fs_backbone(x_s)
                if Fs_bottleneck is not None:
                    z_s_fs = Fs_bottleneck(z_s_fs)

            # ---- Ft forward on source+target
            z_t_ft = Ft_backbone(x_t)
            z_s_ft = Ft_backbone(x_s)
            if Ft_bottleneck is not None:
                z_t_ft = Ft_bottleneck(z_t_ft)
                z_s_ft = Ft_bottleneck(z_s_ft)

            logits_s = C(z_s_ft)
            logits_t = C(z_t_ft)

            # ---- (1) Train D: Fs(xs)=1, Ft(xt)=0
            log_s_D = D(z_s_fs.detach())
            log_t_D = D(z_t_ft.detach())
            ones = torch.ones_like(log_s_D)
            zeros = torch.zeros_like(log_t_D)

            loss_D = 0.5 * (bce(log_s_D, ones).mean() + bce(log_t_D, zeros).mean())
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # ---- (2) weights from D(Fs(xs)) with stop-grad, normalize mean to 1
            with torch.no_grad():
                w = 1.0 - torch.sigmoid(log_s_D)
                w = w / (w.mean() + 1e-8)

            # ---- (3) Adversarial with D0 + GRL
            progress = min(1.0, global_step / total_steps)
            lam = lambda_schedule(progress, lambda_upper=lambda_upper, alpha=alpha, device=device)

            log_s_D0 = D0(z_s_fs.detach())              # Fs side
            log_t_D0 = D0(grad_reverse(z_t_ft, lam))    # Ft side with GRL

            loss_s_D0 = (w * bce(log_s_D0, ones)).mean()
            loss_t_D0 = bce(log_t_D0, zeros).mean()
            loss_adv = 0.5 * (loss_s_D0 + loss_t_D0)

            # ---- (4) Source CE anchor
            loss_cls = ce(logits_s, y_s)

            # ---- (5) Optional target entropy
            loss_ent = torch.tensor(0.0, device=device)
            if entropy_weight > 0:
                p_t = F.softmax(logits_t, dim=1)
                loss_ent = -(p_t * torch.log(p_t + 1e-8)).sum(dim=1).mean()

            # ---- (6) Main update
            loss_main = loss_cls + loss_adv + entropy_weight * loss_ent
            opt_main.zero_grad()
            loss_main.backward()
            opt_main.step()

            global_step += 1

            last = {
                "loss_D": float(loss_D.item()),
                "loss_adv": float(loss_adv.item()),
                "loss_cls": float(loss_cls.item()),
                "loss_ent": float(loss_ent.item()),
                "lam": float(lam),
                "w_mean": float(w.mean().item()),
                "w_std": float(w.std().item()),
                "w_min": float(w.min().item()),
                "w_max": float(w.max().item()),
            }

        print(
            f"[DA] ep {ep+1:03d}/{epochs} "
            f"D={last['loss_D']:.4f} adv={last['loss_adv']:.4f} cls={last['loss_cls']:.4f} "
            f"ent={last['loss_ent']:.4f} lam={last['lam']:.4f} "
            f"w_mean={last['w_mean']:.4f} w_std={last['w_std']:.4f} "
            f"w_min={last['w_min']:.4f} w_max={last['w_max']:.4f}"
        )

    return model


# -------------------------------
# CLI
# -------------------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data_root", type=str, required=True)

    # IMPORTANT: use lowercase to match your folder names (amazon/dslr/webcam)
    p.add_argument("--source", type=str, required=True, choices=["amazon", "dslr", "webcam"])
    p.add_argument("--target", type=str, required=True, choices=["amazon", "dslr", "webcam"])

    p.add_argument("--arch", type=str, default="alexnet", choices=["alexnet"])
    p.add_argument("--bottleneck_dim", type=int, default=256)

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)

    # Paper/table-style: fixed shared-10 list
    p.add_argument("--partial_mode", type=str, default="office_caltech_10", choices=["office_caltech_10"])

    # Stage 1
    p.add_argument("--epochs_cls", type=int, default=200)
    p.add_argument("--lr_cls", type=float, default=1e-3)

    # Stage 2
    p.add_argument("--epochs_da", type=int, default=250)
    p.add_argument("--lr_da", type=float, default=1e-3)
    p.add_argument("--lambda_upper", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--entropy_weight", type=float, default=0.0)

    # Optional: disable ImageNet preload
    p.add_argument("--no_pretrained", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    if args.source == args.target:
        raise ValueError("source and target must be different domains to match A→W, D→W, etc.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    src_train, tgt_train, tgt_test, num_classes, shared_names = make_loaders_optionA(
        data_root=args.data_root,
        source=args.source,
        target=args.target,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        partial_mode=args.partial_mode,
    )

    print("\n== Setup ==")
    print(f"Source domain: {args.source} (train on ALL {num_classes} classes)")
    print(f"Target domain: {args.target} (train unlabeled + test on shared-10 only)")
    print("Shared-10 classes:", shared_names)
    print(f"Source train images: {len(src_train.dataset)}")
    print(f"Target train images (shared-10): {len(tgt_train.dataset)}")
    print(f"Target test images  (shared-10): {len(tgt_test.dataset)}\n")

    model = IWANClassifier(
        num_classes=num_classes,                  # 31-way classifier
        bottleneck_dim=args.bottleneck_dim,
        pretrained=not args.no_pretrained,
    )

    # ---------------- Stage 1: AlexNet+bottleneck baseline ----------------
    print("==> Stage 1: Source training (AlexNet+bottleneck baseline)")
    model = train_source(
        model=model,
        src_train=src_train,
        device=device,
        epochs=args.epochs_cls,
        lr=args.lr_cls,
    )

    # Baseline evaluation on target shared-10
    tgt_acc_pre = eval_acc(model, tgt_test, device)
    print(f"\n[Baseline: AlexNet+bottleneck] tgt_test_acc(shared-10) = {tgt_acc_pre:.4f}\n")

    # ---------------- Stage 2: IWAN (proposed) ----------------
    print("==> Stage 2: IWAN adaptation (unlabeled target train, report target TEST shared-10)")
    model = train_iwan(
        model=model,
        src_train=src_train,
        tgt_train=tgt_train,
        device=device,
        epochs=args.epochs_da,
        lr=args.lr_da,
        lambda_upper=args.lambda_upper,
        alpha=args.alpha,
        entropy_weight=args.entropy_weight,
    )

    tgt_acc_post = eval_acc(model, tgt_test, device)
    print(f"\n[After IWAN] tgt_test_acc(shared-10) = {tgt_acc_post:.4f}\n")

    print("Done.")
    print("Interpretation:")
    print("- Baseline line corresponds to 'AlexNet+bottleneck'")
    print("- IWAN with entropy_weight=0.0 corresponds to 'proposed (γ=0)'")
    print("- IWAN with entropy_weight>0 corresponds to 'proposed' (with target entropy minimization)")


if __name__ == "__main__":
    main()
