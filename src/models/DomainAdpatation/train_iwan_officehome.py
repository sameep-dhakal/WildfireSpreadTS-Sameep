#!/usr/bin/env python3
"""
IWAN on Office-31 (Office-Caltech-10 shared classes) with AlexNet+bottleneck.

Goal: replicate table-style protocol (Option A, standard in DA papers)
- Source train (labeled): ALL source images, ALL 31 classes
- Target train (unlabeled): ALL target images, FILTERED to fixed shared-10 classes
- Target test (evaluation only): ALL target images, same shared-10 classes (labels used ONLY for reporting)

Key FIXES vs your current script:
1) ✅ D0 adversarial loss MUST update Ft using BOTH source and target Ft-features.
   - We compute weights w from frozen Fs(xs).
   - But D0 sees Ft(xs) and Ft(xt) (with GRL), so Ft actually learns alignment.
2) ✅ Proper AlexNet crop size: 227 (common in Office31 AlexNet baselines).
3) ✅ Reasonable training: SGD+momentum, LR scheduler (helps match reported numbers).
4) ✅ Logging shows baseline target acc (shared-10) and post-IWAN target acc (shared-10).

Dataset layout:
DATA_ROOT/
  amazon/<class_name>/*.jpg
  dslr/<class_name>/*.jpg
  webcam/<class_name>/*.jpg
"""

import argparse
import os
import random
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image, UnidentifiedImageError

try:
    import wandb  # type: ignore
except Exception:
    wandb = None



# -------------------------------
# Utils
# -------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_init_wandb(args):
    """Initialize wandb if available and not disabled."""
    if os.environ.get("WANDB_DISABLED") == "1":
        return None
    if wandb is None:
        return None
    project = os.environ.get("WANDB_PROJECT", "office31-partial-da-iwan")
    return wandb.init(project=project, config=vars(args), settings=wandb.Settings(start_method="thread"))


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
# AlexNet feature extractor
# -------------------------------
class AlexNetFeat(nn.Module):
    """Returns a 4096-d feature (fc7 output) from ImageNet pretrained AlexNet."""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = torchvision.models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.alexnet(weights=weights)
        self.features = net.features
        self.avgpool = net.avgpool
        # up to fc7 relu (Dropout, fc6, relu, Dropout, fc7, relu)
        self.classifier_prefix = nn.Sequential(*list(net.classifier.children())[:6])
        self.out_dim = 4096

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier_prefix(x)
        return x


class IWANClassifier(nn.Module):
    """AlexNet feature -> bottleneck -> classifier (31-way)."""
    def __init__(self, num_classes: int, bottleneck_dim: int = 256, pretrained: bool = True):
        super().__init__()
        self.backbone = AlexNetFeat(pretrained=pretrained)
        in_dim = self.backbone.out_dim

        if bottleneck_dim is not None and bottleneck_dim > 0:
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
    """MLP discriminator: in -> 1024 -> 1024 -> 1"""
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
# Data
# -------------------------------


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def is_valid_image_path(p: str) -> bool:
    base = os.path.basename(p)
    if base.startswith("._") or base == ".DS_Store":
        return False
    ext = os.path.splitext(base)[1].lower()
    return ext in IMG_EXTS


class Office31Domain(Dataset):
    def __init__(self, domain_root: str, source_class_to_idx: Dict[str, int], transform):
        self.inner = torchvision.datasets.ImageFolder(domain_root, transform=None)
        self.source_class_to_idx = source_class_to_idx
        self.transform = transform
        self.loader = self.inner.loader

        self.samples: List[Tuple[str, int]] = []
        for path, y_inner in self.inner.samples:
            if not is_valid_image_path(path):
                continue
            cls_name = self.inner.classes[y_inner]
            if cls_name in source_class_to_idx:
                self.samples.append((path, source_class_to_idx[cls_name]))

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid images found under {domain_root}. Check dataset structure.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        try:
            x = self.loader(path)
        except (UnidentifiedImageError, OSError):
            # If something slips through, fall back to another sample instead of crashing the run
            new_idx = (idx + 1) % len(self.samples)
            return self.__getitem__(new_idx)

        if self.transform is not None:
            x = self.transform(x)
        return x, y


class FilterByClassIDs(Subset):
    def __init__(self, base: Office31Domain, keep_ids: List[int]):
        keep = set(int(k) for k in keep_ids)
        indices = [i for i, (_, y) in enumerate(base.samples) if int(y) in keep]
        super().__init__(base, indices)


class SubDataset(Subset):
    """Thin wrapper around torch.utils.data.Subset for clarity."""
    def __init__(self, base: Dataset, indices: List[int]):
        super().__init__(base, indices)


def get_shared10_list(mode: str) -> List[str]:
    """Fixed Office-Caltech-10 shared classes (folder names)."""
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
    src_val_ratio: float,
    split_seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, int, List[str]]:

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # AlexNet-friendly crops (common in Office31 AlexNet baselines)
    train_tf = T.Compose([
        T.Resize(256),
        T.RandomCrop(227),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])
    test_tf = T.Compose([
        T.Resize(256),
        T.CenterCrop(227),
        T.ToTensor(),
        normalize,
    ])

    src_root = os.path.join(data_root, source)
    tgt_root = os.path.join(data_root, target)
    if not os.path.isdir(src_root):
        raise FileNotFoundError(f"Missing source domain folder: {src_root}")
    if not os.path.isdir(tgt_root):
        raise FileNotFoundError(f"Missing target domain folder: {tgt_root}")

    # canonical class mapping from source folder names
    src_probe = torchvision.datasets.ImageFolder(src_root, transform=train_tf)
    source_class_to_idx = src_probe.class_to_idx
    num_classes = len(src_probe.classes)

    # full source (31 classes)
    src_ds_train_tf = Office31Domain(src_root, source_class_to_idx, transform=train_tf)
    src_ds_val_tf   = Office31Domain(src_root, source_class_to_idx, transform=test_tf)

    # shared-10 filtering on target
    shared_names = get_shared10_list(partial_mode)
    missing = [c for c in shared_names if c not in source_class_to_idx]
    if missing:
        raise ValueError(
            f"These shared-10 class folder names are missing in SOURCE '{source}': {missing}\n"
            "Fix by renaming folders OR updating get_shared10_list() to match your dataset naming."
        )
    keep_ids = [source_class_to_idx[c] for c in shared_names]

    tgt_train_full = Office31Domain(tgt_root, source_class_to_idx, transform=train_tf)
    tgt_test_full  = Office31Domain(tgt_root, source_class_to_idx, transform=test_tf)

    # Keep FULL target train (includes outlier classes) for true PDA; evaluate on shared classes only.
    tgt_train_ds = tgt_train_full
    tgt_test_ds  = FilterByClassIDs(tgt_test_full, keep_ids)

    # Split source into train/val for monitoring
    n_src = len(src_ds_train_tf)
    n_val = int(round(n_src * src_val_ratio))
    n_val = min(max(n_val, 1), n_src - 1)  # keep both splits non-empty
    n_train = n_src - n_val
    g = torch.Generator()
    g.manual_seed(split_seed)
    indices = torch.randperm(n_src, generator=g).tolist()
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    src_train_ds = SubDataset(src_ds_train_tf, train_idx)
    src_val_ds   = SubDataset(src_ds_val_tf, val_idx)

    src_train = DataLoader(
        src_train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True, pin_memory=True
    )
    src_val = DataLoader(
        src_val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False, pin_memory=True
    )
    tgt_train = DataLoader(
        tgt_train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True, pin_memory=True
    )
    tgt_test = DataLoader(
        tgt_test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False, pin_memory=True
    )

    return src_train, src_val, tgt_train, tgt_test, num_classes, shared_names


# -------------------------------
# Metrics
# -------------------------------
@torch.no_grad()
def eval_acc(model: IWANClassifier, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
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
    total_loss, total = 0.0, 0
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


def build_optimizer_and_sched(params, lr: float, steps_total: int):
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    # this helps small Office31 stabilize; many baselines use step decay, cosine also works well
    if steps_total <= 0:
        steps_total = 1
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps_total)
    return opt, sched


def train_source(
    model: IWANClassifier,
    src_train: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    src_val: DataLoader,
    wandb_run=None,
) -> IWANClassifier:
    model.to(device)
    ce = nn.CrossEntropyLoss()

    steps_total = epochs * len(src_train)
    opt, sched = build_optimizer_and_sched(model.parameters(), lr=lr, steps_total=steps_total)

    best = float("inf")
    best_state = None

    for ep in range(epochs):
        model.train()
        running, n = 0.0, 0
        for x, y in src_train:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits, _ = model(x)
            loss = ce(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()

            running += loss.item() * y.size(0)
            n += y.size(0)

        train_ce = running / max(1, n)
        # validation on source val/test split
        val_ce = eval_ce(model, src_val, device)
        is_best = val_ce < best
        if is_best:
            best = val_ce
            best_state = deepcopy(model.state_dict())

        print(
            f"[CLS] ep {ep+1:03d}/{epochs} "
            f"src_train_ce={train_ce:.4f} src_val_ce={val_ce:.4f} "
            f"best={best:.4f} lr={sched.get_last_lr()[0]:.6f}"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "stage": "cls",
                    "epoch_cls": ep + 1,
                    "src_train_ce": train_ce,
                    "src_val_ce": val_ce,
                    "src_best_ce": best,
                    "lr_cls": sched.get_last_lr()[0],
                },
                step=ep + 1,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_iwan(
    model: IWANClassifier,
    src_train: DataLoader,
    src_val: DataLoader,
    tgt_train: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    lambda_upper: float,
    alpha: float,
    entropy_weight: float,
):
    """
    IWAN (fixed):
    - Fs = frozen copy of Ft at start of DA stage (backbone + bottleneck)
    - C  = frozen classifier
    - D  = domain disc for importance weights using Fs(xs) vs Ft(xt)
    - D0 = adversarial alignment disc using Ft(xs) vs Ft(xt) with GRL
          source term is weighted by w computed from Fs(xs)
    - Ft updated by: CE(source) + Adv(D0) + entropy_weight * Entropy(target)
    """
    model.to(device)

    # ----- Frozen Fs (copy of current Ft at start of DA stage)
    Fs_backbone = deepcopy(model.backbone).to(device).eval()
    for p in Fs_backbone.parameters():
        p.requires_grad_(False)

    Fs_bottleneck = None
    if model.bottleneck is not None:
        Fs_bottleneck = deepcopy(model.bottleneck).to(device).eval()
        for p in Fs_bottleneck.parameters():
            p.requires_grad_(False)

    # ----- Ft (trainable) + frozen classifier
    Ft_backbone = model.backbone
    Ft_bottleneck = model.bottleneck
    C = model.classifier
    for p in C.parameters():
        p.requires_grad_(False)

    # ----- Discriminators
    feat_dim = model.feat_dim
    D = DomainDiscriminator(in_dim=feat_dim).to(device)
    D0 = DomainDiscriminator(in_dim=feat_dim).to(device)

    bce = nn.BCEWithLogitsLoss(reduction="none")
    ce = nn.CrossEntropyLoss()

    steps_per_epoch = min(len(src_train), len(tgt_train))
    total_steps = max(1, epochs * steps_per_epoch)

    opt_D, sched_D = build_optimizer_and_sched(D.parameters(), lr=lr, steps_total=total_steps)
    # main updates Ft + D0
    main_params = list(Ft_backbone.parameters()) + ([] if Ft_bottleneck is None else list(Ft_bottleneck.parameters())) + list(D0.parameters())
    opt_main, sched_main = build_optimizer_and_sched(main_params, lr=lr, steps_total=total_steps)

    global_step = 0

    best = float("inf")
    best_state = None

    for ep in range(epochs):
        model.train()
        D.train()
        D0.train()

        it_s = iter(src_train)
        it_t = iter(tgt_train)

        last = {}

        for _ in range(steps_per_epoch):
            x_s, y_s = next(it_s)
            x_t, _ = next(it_t)

            x_s = x_s.to(device, non_blocking=True)
            y_s = y_s.to(device, non_blocking=True)
            x_t = x_t.to(device, non_blocking=True)

            # ---- Fs source features (no grad)
            with torch.no_grad():
                z_s_fs = Fs_backbone(x_s)
                if Fs_bottleneck is not None:
                    z_s_fs = Fs_bottleneck(z_s_fs)

            # ---- Ft features on source+target
            z_s_ft = Ft_backbone(x_s)
            z_t_ft = Ft_backbone(x_t)
            if Ft_bottleneck is not None:
                z_s_ft = Ft_bottleneck(z_s_ft)
                z_t_ft = Ft_bottleneck(z_t_ft)

            logits_s = C(z_s_ft)
            logits_t = C(z_t_ft)

            # ============================================================
            # (1) Train D for importance weights
            #     Label: Fs(xs)=1, Ft(xt)=0  (binary domain classification)
            # ============================================================
            log_s_D = D(z_s_fs.detach())
            log_t_D = D(z_t_ft.detach())

            ones = torch.ones_like(log_s_D)
            zeros = torch.zeros_like(log_t_D)

            loss_D = 0.5 * (bce(log_s_D, ones).mean() + bce(log_t_D, zeros).mean())

            opt_D.zero_grad(set_to_none=True)
            loss_D.backward()
            opt_D.step()
            sched_D.step()

            # ---- weights from D(Fs(xs)) stop-grad, normalize mean to 1
            with torch.no_grad():
                w = 1.0 - torch.sigmoid(log_s_D)   # as in IWAN
                w = w / (w.mean() + 1e-8)

            # ============================================================
            # (2) Train Ft + D0 (adversarial alignment)  ✅ FIXED
            #     IMPORTANT:
            #       D0 must see Ft(xs) AND Ft(xt) (with GRL),
            #       otherwise Ft never aligns source features.
            # ============================================================
            progress = min(1.0, global_step / max(1, total_steps))
            lam = lambda_schedule(progress, lambda_upper=lambda_upper, alpha=alpha, device=device)

            # Apply GRL to BOTH domains (standard DANN-style)
            log_s_D0 = D0(grad_reverse(z_s_ft, lam))
            log_t_D0 = D0(grad_reverse(z_t_ft, lam))

            ones0 = torch.ones_like(log_s_D0)
            zeros0 = torch.zeros_like(log_t_D0)

            loss_s_D0 = (w * bce(log_s_D0, ones0)).mean()  # weighted source
            loss_t_D0 = bce(log_t_D0, zeros0).mean()       # target
            loss_adv = 0.5 * (loss_s_D0 + loss_t_D0)

            # ---- source CE anchor
            loss_cls = ce(logits_s, y_s)

            # ---- optional target entropy
            loss_ent = torch.tensor(0.0, device=device)
            if entropy_weight > 0:
                p_t = F.softmax(logits_t, dim=1)
                loss_ent = -(p_t * torch.log(p_t + 1e-8)).sum(dim=1).mean()

            loss_main = loss_cls + loss_adv + entropy_weight * loss_ent

            opt_main.zero_grad(set_to_none=True)
            loss_main.backward()
            opt_main.step()
            sched_main.step()

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
                "lr_main": float(sched_main.get_last_lr()[0]),
            }

        # monitor on source val
        src_ce = eval_ce(model, src_val, device)
        print(
            f"[DA] ep {ep+1:03d}/{epochs} "
            f"D={last['loss_D']:.4f} adv={last['loss_adv']:.4f} cls={last['loss_cls']:.4f} "
            f"ent={last['loss_ent']:.4f} lam={last['lam']:.4f} "
            f"w_mean={last['w_mean']:.4f} w_std={last['w_std']:.4f} "
            f"w_min={last['w_min']:.4f} w_max={last['w_max']:.4f} "
            f"lr={last['lr_main']:.6f} src_val_ce={src_ce:.4f}"
        )

        if src_ce < best:
            best = src_ce
            best_state = deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# -------------------------------
# CLI
# -------------------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--source", type=str, required=True, choices=["amazon", "dslr", "webcam"])
    p.add_argument("--target", type=str, required=True, choices=["amazon", "dslr", "webcam"])

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--src_val_ratio", type=float, default=0.2, help="Source hold-out fraction for validation.")
    p.add_argument("--split_seed", type=int, default=42, help="Seed for source train/val split.")

    p.add_argument("--partial_mode", type=str, default="office_caltech_10", choices=["office_caltech_10"])

    # Stage 1
    p.add_argument("--epochs_cls", type=int, default=200)
    p.add_argument("--lr_cls", type=float, default=1e-3)

    # Stage 2
    p.add_argument("--epochs_da", type=int, default=250)  # use 250 to match table; lower only for quick debug
    p.add_argument("--lr_da", type=float, default=1e-3)
    p.add_argument("--lambda_upper", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=10.0)      # often 10 in DANN-style schedules; 1.0 is very mild
    p.add_argument("--entropy_weight", type=float, default=0.0)

    p.add_argument("--no_pretrained", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    if args.source == args.target:
        raise ValueError("source and target must be different (A→W, D→W, etc).")

    # wandb init (safe even if WANDB_DISABLED=1)
    wandb_run = maybe_init_wandb(args)

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    src_train, src_val, tgt_train, tgt_test, num_classes, shared_names = make_loaders_optionA(
        data_root=args.data_root,
        source=args.source,
        target=args.target,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        partial_mode=args.partial_mode,
        src_val_ratio=args.src_val_ratio,
        split_seed=args.split_seed,
    )

    print("\n== Setup ==")
    print(f"Source domain: {args.source} (train on ALL {num_classes} classes)")
    print(f"Target domain: {args.target} (train unlabeled + test on shared-10 only)")
    print("Shared-10 classes:", shared_names)
    print(f"Source train images: {len(src_train.dataset)}")
    print(f"Source val images: {len(src_val.dataset)}")
    print(f"Target train images (full target, includes outlier classes): {len(tgt_train.dataset)}")
    print(f"Target test images  (shared-10): {len(tgt_test.dataset)}\n")

    model = IWANClassifier(
        num_classes=num_classes,
        bottleneck_dim=256,
        pretrained=not args.no_pretrained,
    )

    # ---------------- Stage 1 baseline ----------------
    print("==> Stage 1: Source training (AlexNet+bottleneck baseline)")
    model = train_source(
        model=model,
        src_train=src_train,
        src_val=src_val,
        device=device,
        epochs=args.epochs_cls,
        lr=args.lr_cls,
        wandb_run=wandb_run,
    )

    tgt_acc_pre = eval_acc(model, tgt_test, device)
    print(f"\n[Baseline: AlexNet+bottleneck] tgt_test_acc(shared-10) = {tgt_acc_pre:.4f}\n")

    # ---------------- Stage 2 IWAN ----------------
    print("==> Stage 2: IWAN adaptation (unlabeled target train, report target TEST shared-10)")
    model = train_iwan(
        model=model,
        src_train=src_train,
        src_val=src_val,
        tgt_train=tgt_train,
        device=device,
        epochs=args.epochs_da,
        lr=args.lr_da,
        lambda_upper=args.lambda_upper,
        alpha=args.alpha,
        entropy_weight=args.entropy_weight,
        wandb_run=wandb_run,
    )

    tgt_acc_post = eval_acc(model, tgt_test, device)
    print(f"\n[After IWAN] tgt_test_acc(shared-10) = {tgt_acc_post:.4f}\n")

    print("Interpretation:")
    print("- Baseline line corresponds to 'AlexNet+bottleneck'")
    print("- IWAN with entropy_weight=0.0 corresponds to 'proposed (γ=0)'")
    print("- IWAN with entropy_weight>0 corresponds to 'proposed' (with target entropy minimization)")

    if wandb_run is not None:
        wandb_run.log(
            {
                "src_val_ce_best": getattr(model, "src_best_ce", None),
                "tgt_test_acc_pre": tgt_acc_pre,
                "tgt_test_acc_post": tgt_acc_post,
            }
        )
        wandb_run.finish()


if __name__ == "__main__":
    main()
