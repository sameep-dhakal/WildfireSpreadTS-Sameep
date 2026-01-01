#!/usr/bin/env python3
"""
IWAN (Importance Weighted Adversarial Nets) for Partial Domain Adaptation on Office-31.

Goal: replicate the Office-31 table setting like A31->W10, D31->W10, etc.
- Source domain has 31 classes (A31/W31/D31).
- Target domain is restricted to a shared subset of K classes (default K=10, "Office-10").
- Train with unlabeled target train split, evaluate on target TEST split (shared classes only).

Dataset format you showed (works):
office31/
  amazon/
    class_name/*.jpg
  webcam/
    class_name/*.jpg
  dslr/
    class_name/*.jpg

Splits:
- If official list files exist, we will use them:
    list_root/
      amazon_train.txt, amazon_test.txt
      webcam_train.txt, webcam_test.txt
      dslr_train.txt,  dslr_test.txt
  Each line: "relative_path label"
  where relative_path is relative to data_root (e.g., "amazon/back_pack/img_0001.jpg").

- If list files do NOT exist, we auto-generate deterministic splits and WRITE them into list_root.

Models supported:
- AlexNet + bottleneck (paper baseline)  [default]
- ResNet18 / ResNet50 (optional)

Two stages:
1) Source pretrain on source train, early stop on source test (or source val if you prefer).
2) IWAN adaptation:
   - Freeze Fs (copy of feature extractor) and C (classifier).
   - Train D to estimate importance weights on source (using Fs(x_s) vs Ft(x_t)).
   - Train D0 + Ft for weighted adversarial alignment.
   - Optional target entropy minimization (gamma = entropy_weight).

IMPORTANT: For table replication, you should run:
A->W, D->W, W->D, A->D, D->A, W->A
and report target TEST accuracy.

Example:
python train_iwan_office31.py \
  --data_root /develop/data/office31 \
  --list_root /develop/data/office31/image_list \
  --source amazon --target webcam \
  --arch alexnet --bottleneck_dim 256 \
  --shared_k 10 --shared_seed 0 \
  --epochs_cls 50 --epochs_da 100 \
  --batch_size 64 --lr_cls 0.001 --lr_da 0.001 \
  --lambda_upper 0.1 --alpha 10.0 \
  --entropy_weight 0.0

For "proposed (gamma>0)" set entropy_weight (e.g., 0.1).
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
from torch.utils.data import DataLoader, Dataset
from PIL import ImageFile

# Allow PIL to load truncated images rather than crashing a worker
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import wandb  # type: ignore
except Exception:
    wandb = None


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
# Utils
# -------------------------------
def norm_name(s: str) -> str:
    # normalize class names to match across datasets with small naming differences
    return (
        s.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("__", "_")
    )


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def set_lr(opt: torch.optim.Optimizer, lr: float):
    for pg in opt.param_groups:
        pg["lr"] = lr


def maybe_init_wandb(args):
    """
    Start a wandb run if the package is available and not disabled.
    """
    if os.environ.get("WANDB_DISABLED") == "1":
        return None
    if wandb is None:
        print("[INFO] wandb not available; skipping wandb logging.")
        return None
    project = os.environ.get("WANDB_PROJECT", "office31-partial-da-iwan")
    run = wandb.init(
        project=project,
        config=vars(args),
        name=f"{args.source}->{args.target}_ew{args.entropy_weight}",
        settings=wandb.Settings(start_method="thread"),
    )
    return run


def lambda_schedule(progress: float, lambda_upper: float, alpha: float, device: torch.device) -> float:
    """
    Common DA ramp: 2U/(1+exp(-alpha*p)) - U
    Note: In many papers alpha ~ 10 for a sharper ramp. You were using 1.0, which is very slow.
    """
    p = torch.tensor(float(progress), device=device)
    val = 2.0 * lambda_upper / (1.0 + torch.exp(-alpha * p)) - lambda_upper
    return float(val.item())


def inv_lr(base_lr: float, progress: float, gamma: float = 10.0, power: float = 0.75) -> float:
    """
    Standard DA LR decay: lr = lr0 / (1 + gamma * p) ** power, with p in [0,1].
    Matches the schedule used in many Office-31 PDA baselines.
    """
    return float(base_lr / ((1.0 + gamma * progress) ** power))


# -------------------------------
# List-file dataset
# -------------------------------
class OfficeListDataset(Dataset):
    """
    Reads list files of form:
        relative_path label
    where relative_path is relative to data_root, like "amazon/back_pack/img_0001.jpg".
    If label is -1, we ignore it (useful for unlabeled target train).
    """
    def __init__(self, data_root: str, list_file: str, transform, class_to_idx: Dict[str, int]):
        self.data_root = data_root
        self.transform = transform
        self.class_to_idx = class_to_idx

        items: List[Tuple[str, int]] = []
        with open(list_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                rel = parts[0]
                y = int(parts[1]) if len(parts) > 1 else -1
                abs_path = os.path.join(data_root, rel)

                # We trust the label in list file IF it matches our canonical mapping.
                # But we also support rebuilding from folder name if needed.
                if y >= 0:
                    items.append((abs_path, y))
                else:
                    items.append((abs_path, -1))

        self.items = items
        self.loader = torchvision.datasets.folder.default_loader

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, y = self.items[idx]
        try:
            x = self.loader(path)
        except OSError as e:
            # Skip corrupted/truncated images gracefully
            print(f"[WARN] Skipping corrupted image: {path} ({e})")
            # Simple fallback: return a zero tensor with correct shape; label stays y
            x = torch.zeros(3, 224, 224)
        if self.transform is not None:
            x = self.transform(x)
        return x, y


def scan_domain_classes(domain_root: str) -> List[str]:
    classes = []
    for name in os.listdir(domain_root):
        p = os.path.join(domain_root, name)
        if os.path.isdir(p):
            classes.append(name)
    classes.sort()
    return classes


def build_canonical_mapping(data_root: str, source_domain: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Canonical label mapping is taken from SOURCE domain folders.
    This is critical so labels are consistent across domains.
    """
    src_root = os.path.join(data_root, source_domain)
    classes = scan_domain_classes(src_root)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    return class_to_idx, idx_to_class


def get_default_office10_names() -> List[str]:
    """
    Common Office-10 shared class set used in PDA papers.
    Real datasets sometimes use slightly different spellings.
    We match using normalized names.
    """
    return [
        "back_pack",
        "calculator",
        "headphones",
        "keyboard",
        "laptop_computer",
        "monitor",
        "mouse",
        "mug",
        "projector",
        "speaker",
    ]


def resolve_shared_ids(
    idx_to_class: Dict[int, str],
    shared_names: List[str],
    shared_k: int,
    shared_seed: int,
) -> Tuple[List[int], List[str]]:
    """
    Prefer matching the known shared_names list if possible.
    If not found (name mismatch), we fall back to random K classes (deterministic),
    but then you should NOT expect to match the paper's table.
    """
    # Build normalized lookup for source classes
    src_norm_to_id: Dict[str, int] = {}
    for i, cname in idx_to_class.items():
        src_norm_to_id[norm_name(cname)] = i

    matched_ids = []
    matched_names = []
    for nm in shared_names:
        key = norm_name(nm)
        if key in src_norm_to_id:
            matched_ids.append(src_norm_to_id[key])
            matched_names.append(idx_to_class[src_norm_to_id[key]])

    if len(matched_ids) >= shared_k:
        matched_ids = matched_ids[:shared_k]
        matched_names = matched_names[:shared_k]
        return sorted(matched_ids), matched_names

    # fallback: deterministic random K over source classes
    rng = np.random.default_rng(shared_seed)
    num_classes = len(idx_to_class)
    keep_ids = sorted(rng.choice(num_classes, size=shared_k, replace=False).tolist())
    keep_names = [idx_to_class[i] for i in keep_ids]
    print(
        "[WARN] Could not match enough shared class names in your folder naming.\n"
        "       Falling back to RANDOM shared classes. This will NOT replicate the paper table.\n"
        "       Fix by renaming folders to match Office-10 names or pass --shared_classes explicitly."
    )
    return keep_ids, keep_names


def write_split_lists_if_missing(
    data_root: str,
    list_root: str,
    domain: str,
    class_to_idx: Dict[str, int],
    test_ratio: float,
    split_seed: int,
):
    """
    If {domain}_train.txt and {domain}_test.txt do not exist, create them deterministically.
    """
    ensure_dir(list_root)
    tr_file = os.path.join(list_root, f"{domain}_train.txt")
    te_file = os.path.join(list_root, f"{domain}_test.txt")
    if os.path.exists(tr_file) and os.path.exists(te_file):
        return

    print(
        "[WARN] Official list files not found; auto-splitting this run. "
        "This will NOT match the paper numbers. Provide the Office-31 list files in list_root to replicate Table 3."
    )
    domain_root = os.path.join(data_root, domain)
    # Collect all image paths
    all_items: List[Tuple[str, int]] = []
    for cls in sorted(class_to_idx.keys()):
        cls_dir = os.path.join(domain_root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fn in os.listdir(cls_dir):
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                rel = os.path.join(domain, cls, fn)
                y = class_to_idx[cls]
                all_items.append((rel, y))

    # Deterministic shuffle and split
    rng = np.random.default_rng(split_seed)
    rng.shuffle(all_items)
    n = len(all_items)
    n_test = int(round(n * test_ratio))
    test_items = all_items[:n_test]
    train_items = all_items[n_test:]

    with open(tr_file, "w") as f:
        for rel, y in train_items:
            f.write(f"{rel} {y}\n")
    with open(te_file, "w") as f:
        for rel, y in test_items:
            f.write(f"{rel} {y}\n")

    print(f"[INFO] Wrote split lists: {tr_file} ({len(train_items)}), {te_file} ({len(test_items)})")


def filter_list_file(
    in_file: str,
    out_file: str,
    keep_ids: List[int],
):
    """
    Write a filtered list file that keeps only lines whose label is in keep_ids.
    """
    keep = set(int(k) for k in keep_ids)
    ensure_dir(os.path.dirname(out_file))
    with open(in_file, "r") as fin, open(out_file, "w") as fout:
        for line in fin:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            rel, y = parts[0], int(parts[1])
            if y in keep:
                fout.write(f"{rel} {y}\n")


@dataclass
class DataPack:
    src_train: DataLoader
    src_test: DataLoader
    tgt_train: DataLoader
    tgt_test: DataLoader
    num_classes: int
    keep_ids: List[int]
    keep_names: List[str]


def make_loaders_office31(
    data_root: str,
    list_root: str,
    source: str,
    target: str,
    batch_size: int,
    num_workers: int,
    test_ratio: float,
    split_seed: int,
    shared_k: int,
    shared_seed: int,
    shared_classes: Optional[str] = None,
) -> DataPack:
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

    class_to_idx, idx_to_class = build_canonical_mapping(data_root, source)
    num_classes = len(class_to_idx)

    # Determine shared class IDs
    if shared_classes is not None and shared_classes.strip():
        names = [x.strip() for x in shared_classes.split(",") if x.strip()]
    else:
        names = get_default_office10_names()
    keep_ids, keep_names = resolve_shared_ids(idx_to_class, names, shared_k=shared_k, shared_seed=shared_seed)

    # Ensure split lists exist for both domains (if official lists missing)
    write_split_lists_if_missing(data_root, list_root, source, class_to_idx, test_ratio=test_ratio, split_seed=split_seed)
    write_split_lists_if_missing(data_root, list_root, target, class_to_idx, test_ratio=test_ratio, split_seed=split_seed)

    src_train_list = os.path.join(list_root, f"{source}_train.txt")
    src_test_list  = os.path.join(list_root, f"{source}_test.txt")
    tgt_train_list = os.path.join(list_root, f"{target}_train.txt")
    tgt_test_list  = os.path.join(list_root, f"{target}_test.txt")

    # Filter target TRAIN/TEST to shared IDs.
    # PDA benchmark assumes target label space is a subset (Office-10), so target data
    # should only contain the shared classes.
    filt_root = os.path.join(list_root, "filtered_shared")
    tgt_train_f = os.path.join(filt_root, f"{target}_train_shared{shared_k}.txt")
    tgt_test_f  = os.path.join(filt_root, f"{target}_test_shared{shared_k}.txt")
    if not (os.path.exists(tgt_train_f) and os.path.exists(tgt_test_f)):
        filter_list_file(tgt_train_list, tgt_train_f, keep_ids)
        filter_list_file(tgt_test_list,  tgt_test_f,  keep_ids)

    # Datasets
    src_train_ds = OfficeListDataset(data_root, src_train_list, train_tf, class_to_idx)
    src_test_ds  = OfficeListDataset(data_root, src_test_list,  test_tf,  class_to_idx)

    tgt_train_ds = OfficeListDataset(data_root, tgt_train_f, train_tf, class_to_idx)
    tgt_test_ds  = OfficeListDataset(data_root, tgt_test_f,  test_tf,  class_to_idx)

    src_train = DataLoader(src_train_ds, batch_size=batch_size, shuffle=True,
                           num_workers=num_workers, drop_last=True, pin_memory=True)
    src_test  = DataLoader(src_test_ds,  batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, drop_last=False, pin_memory=True)

    tgt_train = DataLoader(tgt_train_ds, batch_size=batch_size, shuffle=True,
                           num_workers=num_workers, drop_last=True, pin_memory=True)
    tgt_test  = DataLoader(tgt_test_ds,  batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, drop_last=False, pin_memory=True)

    return DataPack(
        src_train=src_train,
        src_test=src_test,
        tgt_train=tgt_train,
        tgt_test=tgt_test,
        num_classes=num_classes,
        keep_ids=keep_ids,
        keep_names=keep_names,
    )


# -------------------------------
# Models
# -------------------------------
class AlexNetBackbone(nn.Module):
    """
    Torchvision AlexNet:
      features -> avgpool -> flatten -> classifier[0:6] gives 4096-d (fc7 output after ReLU+Dropout).
    """
    def __init__(self):
        super().__init__()
        net = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
        self.features = net.features
        self.avgpool = net.avgpool
        self.fc = nn.Sequential(*list(net.classifier.children())[:6])  # up to fc7 activation
        self.out_dim = 4096

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResNetBackbone(nn.Module):
    def __init__(self, arch: str):
        super().__init__()
        if arch == "resnet50":
            net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        elif arch == "resnet18":
            net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported resnet arch: {arch}")
        self.out_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.net = net

    def forward(self, x):
        return self.net(x)


class IWANClassifier(nn.Module):
    """
    Backbone + bottleneck + classifier.
    For paper-style AlexNet+bottleneck: bottleneck_dim=256.
    """
    def __init__(self, num_classes: int, arch: str = "alexnet", bottleneck_dim: int = 256):
        super().__init__()
        self.arch = arch
        if arch == "alexnet":
            self.backbone = AlexNetBackbone()
        elif arch in ("resnet18", "resnet50"):
            self.backbone = ResNetBackbone(arch)
        else:
            raise ValueError(f"Unsupported arch: {arch}")

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
    """
    MLP discriminator used in many PDA baselines (1024, 1024, 1).
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


def freeze_bn(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)


# -------------------------------
# Metrics
# -------------------------------
@torch.no_grad()
def eval_acc(model: IWANClassifier, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
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
    total_loss, total = 0.0, 0
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
def train_source(
    model: IWANClassifier,
    src_train: DataLoader,
    src_test: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    patience: int,
    wandb_run=None,
):
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    ce = nn.CrossEntropyLoss()

    best = float("inf")
    best_state = None
    bad = 0
    steps_per_epoch = len(src_train)
    total_steps = max(1, epochs * steps_per_epoch)
    global_step = 0

    for ep in range(epochs):
        model.train()
        for x, y in src_train:
            progress = min(1.0, global_step / total_steps)
            lr_cur = inv_lr(lr, progress)
            set_lr(opt, lr_cur)

            x = x.to(device)
            y = y.to(device)
            logits, _ = model(x)
            loss = ce(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            global_step += 1

        te_loss = eval_ce(model, src_test, device)
        if te_loss < best:
            best = te_loss
            best_state = deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1

        print(f"[CLS] ep {ep+1}/{epochs} src_test_ce={te_loss:.4f} best={best:.4f}")
        if wandb_run is not None:
            wandb_run.log(
                {
                    "stage": "cls",
                    "epoch_cls": ep + 1,
                    "src_test_ce": te_loss,
                    "src_best_ce": best,
                    "lr_cls": lr_cur,
                },
                step=ep + 1,
            )
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
    src_test: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    lambda_upper: float,
    alpha: float,
    entropy_weight: float,
    patience: int,
    freeze_bn_flag: bool,
    wandb_run=None,
):
    """
    Stage 2 IWAN:
    - Fs = frozen copy of backbone+bottleneck (source feature extractor)
    - Ft = current backbone+bottleneck (trainable)
    - C  = frozen classifier
    - D  = weight discriminator
    - D0 = adversarial discriminator for feature alignment
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

    # Trainable Ft + D0
    Ft_backbone = model.backbone
    Ft_bottleneck = model.bottleneck
    C = model.classifier
    for p in C.parameters():
        p.requires_grad_(False)

    if freeze_bn_flag:
        freeze_bn(Ft_backbone)

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

        last_stats = {}

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
            z_s_ft = Ft_backbone(x_s)
            z_t_ft = Ft_backbone(x_t)
            if Ft_bottleneck is not None:
                z_s_ft = Ft_bottleneck(z_s_ft)
                z_t_ft = Ft_bottleneck(z_t_ft)

            logits_s = C(z_s_ft)
            logits_t = C(z_t_ft)

            # 3) Train D: source=1, target=0 (for weights)
            log_s_D = D(z_s_fs.detach())
            log_t_D = D(z_t_ft.detach())

            ones = torch.ones_like(log_s_D)
            zeros = torch.zeros_like(log_t_D)

            loss_D = 0.5 * (bce(log_s_D, ones).mean() + bce(log_t_D, zeros).mean())
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # 4) Importance weights from D(Fs(x_s)) (stop-grad), mean-normalize
            with torch.no_grad():
                w = 1.0 - torch.sigmoid(log_s_D)
                w = w / (w.mean() + 1e-8)

            # 5) Adversarial alignment via D0 with GRL
            progress = min(1.0, global_step / max(1, epochs * steps_per_epoch))
            lr_cur = inv_lr(lr, progress)
            set_lr(opt_D, lr_cur)
            set_lr(opt_main, lr_cur)
            lam = lambda_schedule(progress, lambda_upper=lambda_upper, alpha=alpha, device=device)

            log_s_D0 = D0(z_s_fs.detach())                 # source (frozen features)
            log_t_D0 = D0(grad_reverse(z_t_ft, lam))       # target (Ft gets reversed gradients)

            loss_s_D0 = (w * bce(log_s_D0, ones)).mean()
            loss_t_D0 = bce(log_t_D0, zeros).mean()
            loss_adv = 0.5 * (loss_s_D0 + loss_t_D0)

            # 6) Source CE anchor (keeps classifier meaningful)
            loss_cls = ce(logits_s, y_s)

            # 7) Optional target entropy (gamma)
            loss_ent = torch.tensor(0.0, device=device)
            if entropy_weight > 0:
                p_t = F.softmax(logits_t, dim=1)
                loss_ent = -(p_t * torch.log(p_t + 1e-8)).sum(dim=1).mean()

            # 8) Main update
            loss_main = loss_cls + loss_adv + entropy_weight * loss_ent
            opt_main.zero_grad()
            loss_main.backward()
            opt_main.step()

            global_step += 1

            last_stats = {
                "loss_D": float(loss_D.item()),
                "loss_adv": float(loss_adv.item()),
                "loss_cls": float(loss_cls.item()),
                "loss_ent": float(loss_ent.item()),
                "lam": float(lam),
                "lr": float(lr_cur),
                "w_mean": float(w.mean().item()),
                "w_std": float(w.std().item()),
                "w_min": float(w.min().item()),
                "w_max": float(w.max().item()),
            }

        # monitor using source test CE (no target labels used)
        src_ce = eval_ce(model, src_test, device)
        print(
            f"[DA] ep {ep+1}/{epochs} src_test_ce={src_ce:.4f} "
            f"D={last_stats.get('loss_D',0):.4f} adv={last_stats.get('loss_adv',0):.4f} "
            f"cls={last_stats.get('loss_cls',0):.4f} ent={last_stats.get('loss_ent',0):.4f} "
            f"lam={last_stats.get('lam',0):.4f} "
            f"w_mean={last_stats.get('w_mean',0):.4f} w_std={last_stats.get('w_std',0):.4f} "
            f"w_min={last_stats.get('w_min',0):.4f} w_max={last_stats.get('w_max',0):.4f}"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "stage": "da",
                    "epoch_da": ep + 1,
                    "src_test_ce": src_ce,
                    "loss_D": last_stats.get("loss_D", 0.0),
                    "loss_adv": last_stats.get("loss_adv", 0.0),
                    "loss_cls": last_stats.get("loss_cls", 0.0),
                    "loss_ent": last_stats.get("loss_ent", 0.0),
                    "lambda": last_stats.get("lam", 0.0),
                    "lr_da": last_stats.get("lr", 0.0),
                    "w_mean": last_stats.get("w_mean", 0.0),
                    "w_std": last_stats.get("w_std", 0.0),
                    "w_min": last_stats.get("w_min", 0.0),
                    "w_max": last_stats.get("w_max", 0.0),
                    "entropy_weight": entropy_weight,
                },
                step=ep + 1,
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
    p.add_argument("--data_root", type=str, required=True, help="Path to office31 root (contains amazon/webcam/dslr).")
    p.add_argument("--list_root", type=str, required=True, help="Where split list files live (or will be written).")
    p.add_argument("--source", type=str, required=True, choices=["amazon", "webcam", "dslr"])
    p.add_argument("--target", type=str, required=True, choices=["amazon", "webcam", "dslr"])

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)

    # splits (used ONLY if official lists are missing)
    p.add_argument("--test_ratio", type=float, default=0.3, help="Auto-split test ratio if list files missing.")
    p.add_argument("--split_seed", type=int, default=42)

    # partial DA shared set
    p.add_argument("--shared_k", type=int, default=10, help="Target uses K shared classes (default 10).")
    p.add_argument("--shared_seed", type=int, default=0, help="Seed if fallback random shared classes needed.")
    p.add_argument(
        "--shared_classes",
        type=str,
        default="",
        help="Comma-separated shared class names to match your folder names (overrides default Office-10 list).",
    )

    # model
    p.add_argument("--arch", type=str, default="alexnet", choices=["alexnet", "resnet18", "resnet50"])
    p.add_argument("--bottleneck_dim", type=int, default=256)
    p.add_argument("--freeze_bn", type=int, default=0, help="Use 1 for ResNet stability; not needed for AlexNet.")

    # stage 1
    p.add_argument("--epochs_cls", type=int, default=50)
    p.add_argument("--lr_cls", type=float, default=1e-3)
    p.add_argument("--patience_cls", type=int, default=10)

    # stage 2
    p.add_argument("--epochs_da", type=int, default=100)
    p.add_argument("--lr_da", type=float, default=1e-3)
    p.add_argument("--patience_da", type=int, default=0)

    p.add_argument("--lambda_upper", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=10.0, help="GRL schedule sharpness; 10 is common.")
    p.add_argument("--entropy_weight", type=float, default=0.0, help="gamma (0 for proposed(gamma=0)).")

    return p.parse_args()


def main():
    args = parse_args()
    wandb_run = maybe_init_wandb(args)
    if args.source == args.target:
        print("[SKIP] source and target are the same; skipping run.")
        if wandb_run is not None:
            wandb_run.log({"skipped_same_domain": 1})
            wandb_run.finish()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    data = make_loaders_office31(
        data_root=args.data_root,
        list_root=args.list_root,
        source=args.source,
        target=args.target,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
        shared_k=args.shared_k,
        shared_seed=args.shared_seed,
        shared_classes=args.shared_classes,
    )

    print("\n== Partial DA setup (Office-31 -> Office-10) ==")
    print(f"Source total classes: {data.num_classes}")
    print(f"Target shared classes K: {len(data.keep_ids)}")
    print("Shared class names:", data.keep_names)

    model = IWANClassifier(
        num_classes=data.num_classes,
        arch=args.arch,
        bottleneck_dim=args.bottleneck_dim,
    )

    # ---------------- Stage 1: source only ----------------
    print("\n==> Stage 1: Source pretrain (source-train -> source-test monitoring)")
    model = train_source(
        model=model,
        src_train=data.src_train,
        src_test=data.src_test,
        device=device,
        epochs=args.epochs_cls,
        lr=args.lr_cls,
        patience=args.patience_cls,
        wandb_run=wandb_run,
    )

    src_test_acc = eval_acc(model, data.src_test, device)
    tgt_test_acc_pre = eval_acc(model, data.tgt_test, device)
    print(f"[After CLS] src_test_acc={src_test_acc:.4f}  tgt_test_acc(shared only)={tgt_test_acc_pre:.4f}")
    if wandb_run is not None:
        wandb_run.log(
            {
                "src_test_acc_pre": src_test_acc,
                "tgt_test_acc_pre": tgt_test_acc_pre,
                "entropy_weight": args.entropy_weight,
            }
        )

    # ---------------- Stage 2: IWAN ----------------
    print("\n==> Stage 2: IWAN adaptation (unlabeled target train, report target TEST)")
    model = train_iwan(
        model=model,
        src_train=data.src_train,
        tgt_train=data.tgt_train,
        src_test=data.src_test,
        device=device,
        epochs=args.epochs_da,
        lr=args.lr_da,
        lambda_upper=args.lambda_upper,
        alpha=args.alpha,
        entropy_weight=args.entropy_weight,
        patience=args.patience_da,
        freeze_bn_flag=bool(args.freeze_bn),
        wandb_run=wandb_run,
    )

    tgt_test_acc_post = eval_acc(model, data.tgt_test, device)
    print(f"[After IWAN] tgt_test_acc(shared only)={tgt_test_acc_post:.4f}")

    print("\n=== SUMMARY ===")
    print(f"Transfer: {args.source} -> {args.target} (K={args.shared_k})")
    print(f"Baseline (AlexNet+bottleneck) target TEST: {tgt_test_acc_pre*100:.2f}%")
    print(f"IWAN target TEST: {tgt_test_acc_post*100:.2f}%")
    print(f"Entropy weight (gamma): {args.entropy_weight}")
    if wandb_run is not None:
        wandb_run.log(
            {
                "tgt_test_acc_post": tgt_test_acc_post,
                "baseline_tgt_test_pct": tgt_test_acc_pre * 100.0,
                "iwan_tgt_test_pct": tgt_test_acc_post * 100.0,
                "entropy_weight": args.entropy_weight,
            }
        )
        wandb_run.finish()


if __name__ == "__main__":
    main()
