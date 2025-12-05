# import os
# from typing import Optional, Tuple

# import torch
# import torch.nn as nn
# import segmentation_models_pytorch as smp

# from ..BaseModel import BaseModel


# # ============================================================
# # DOMAIN HEAD (DISCRIMINATOR)
# # ============================================================
# class DomainLogitHead(nn.Module):
#     """Small stable MLP: pooled feature map → logit."""
#     def __init__(self, in_channels: int):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),   # (B, C, H, W) → (B, C, 1, 1)
#             nn.Flatten(),              # (B, C)
#             nn.Linear(in_channels, 512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.1),
#             nn.Linear(512, 128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.1),
#             nn.Linear(128, 1),
#         )

#     def forward(self, x):
#         return self.net(x).squeeze(1)
        

# # ============================================================
# # IWAN STAGE 2 WEIGHT ESTIMATOR
# # ============================================================
# class IWANStage2_WeightEstimator(BaseModel):
#     """
#     IWAN Stage-2:
#         - Load encoder Fs from Stage-1 (UNet)
#         - Freeze encoder (including disabling BN updates)
#         - Train only a small domain classifier D
#         - No segmentation
#         - No validation
#         - Checkpoint monitors: score_D_epoch (min)
#     """

#     def __init__(
#         self,
#         encoder_name: str = "resnet18",
#         encoder_weights: Optional[str] = None,
#         in_channels: Optional[int] = None,
#         n_channels: int = 7,
#         flatten_temporal_dimension: bool = True,
#         pos_class_weight: float = 1.0,
#         loss_function: str = "BCE",
#         use_doy: bool = False,
#         crop_before_eval: bool = False,
#         required_img_size: Optional[Tuple[int, int]] = None,
#         alpha_focal: Optional[float] = None,
#         f1_threshold: Optional[float] = None,
#         ckpt_dir: str = None,
#         **kwargs,
#     ):
#         # allow in_channels override
#         if in_channels is not None:
#             n_channels = in_channels

#         # initialize BaseModel (Lightning compatible)
#         super().__init__(
#             n_channels=n_channels,
#             flatten_temporal_dimension=flatten_temporal_dimension,
#             pos_class_weight=pos_class_weight,
#             loss_function=loss_function,
#             use_doy=use_doy,
#             crop_before_eval=crop_before_eval,
#             required_img_size=required_img_size,
#             alpha_focal=alpha_focal,
#             f1_threshold=f1_threshold,
#             **kwargs,
#         )

#         # save all hparams
#         self.save_hyperparameters()

#         # ---------------------------------------------
#         # LOAD STAGE-1 CHECKPOINT
#         # ---------------------------------------------
#         if ckpt_dir is None or not os.path.isdir(ckpt_dir):
#             raise FileNotFoundError(f"❌ Stage-1 ckpt directory not found: {ckpt_dir}")

#         ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
#         if not ckpt_files:
#             raise FileNotFoundError(f"❌ No .ckpt files found in {ckpt_dir}")

#         ckpt_path = os.path.join(ckpt_dir, ckpt_files[0])
#         print(f"✅ Loading Stage-1 encoder from: {ckpt_path}")

#         stage1 = smp.Unet(
#             encoder_name=encoder_name,
#             encoder_weights=None,
#             in_channels=n_channels,
#             classes=1,
#         )

#         ckpt = torch.load(ckpt_path, map_location="cpu")
#         sd = ckpt.get("state_dict", ckpt)

#         # load encoder weights only
#         stage1.load_state_dict(
#             {k.replace("model.", ""): v for k, v in sd.items() if "encoder" in k},
#             strict=False
#         )

#         # ---------------------------------------------
#         # CORRECT MODULE REGISTRATION
#         # ---------------------------------------------
#         self.Fs = stage1.encoder            # register module
#         self.Fs.eval()                      # set eval mode

#         # Fully freeze encoder parameters
#         for p in self.Fs.parameters():
#             p.requires_grad_(False)

#         # Disable BN update
#         for m in self.Fs.modules():
#             if isinstance(m, nn.BatchNorm2d):
#                 m.track_running_stats = False
#                 m.running_mean = None
#                 m.running_var = None

#         # ---------------------------------------------
#         # DOMAIN DISCRIMINATOR
#         # ---------------------------------------------
#         feat_dim = self.Fs.out_channels[-1]
#         self.D = DomainLogitHead(feat_dim)
#         self.bce = nn.BCEWithLogitsLoss()

#         # iterator for target batches
#         self.target_iter = None

#     # ============================================================
#     # DISCRIMINATOR FORWARD
#     # ============================================================
#     def forward_D(self, x_s, x_t):
#         # flatten T dimension if needed
#         if self.hparams.flatten_temporal_dimension and x_s.ndim == 5:
#             x_s = x_s.flatten(1, 2)
#         if self.hparams.flatten_temporal_dimension and x_t.ndim == 5:
#             x_t = x_t.flatten(1, 2)

#         with torch.no_grad():                      # encoder NEVER updates
#             f_s = self.Fs(x_s)[-1]
#             f_t = self.Fs(x_t)[-1]

#         log_s = self.D(f_s)
#         log_t = self.D(f_t)

#         y_s = torch.ones_like(log_s)
#         y_t = torch.zeros_like(log_t)

#         loss = 0.5 * self.bce(log_s, y_s) + 0.5 * self.bce(log_t, y_t)

#         with torch.no_grad():
#             p_s = torch.sigmoid(log_s)
#             p_t = torch.sigmoid(log_t)
#             acc = 0.5 * ((p_s > 0.5).float().mean() + (p_t < 0.5).float().mean())

#         score = loss.detach()

#         return loss, acc, score

#     # ============================================================
#     # TRAINING LOOP
#     # ============================================================
#     def on_train_start(self):
#         dm = self.trainer.datamodule
#         self.target_iter = iter(dm.target_dataloader())
#         print("🟢 Stage-2: initialized target iterator")

#     def training_step(self, batch, _):
#         x_s, _ = batch

#         try:
#             x_t, _ = next(self.target_iter)
#         except StopIteration:
#             self.target_iter = iter(self.trainer.datamodule.target_dataloader())
#             x_t, _ = next(self.target_iter)

#         x_s = x_s.to(self.device)
#         x_t = x_t.to(self.device)

#         loss, acc, score = self.forward_D(x_s, x_t)

#         self.log("train_loss_D", loss, on_epoch=True, prog_bar=True)
#         self.log("train_domain_acc", acc, on_epoch=True, prog_bar=True)
#         self.log("score_D", score, on_epoch=True, prog_bar=False)

#         return loss

#     # ============================================================
#     # NO VALIDATION
#     # ============================================================
#     def validation_step(self, *args, **kwargs):
#         return None

#     # ============================================================
#     # CORRECT OPTIMIZER (NOW NOT OVERRIDDEN BY CLI)
#     # ============================================================
#     def configure_optimizers(self):
#         return torch.optim.Adam(self.D.parameters(), lr=5e-5)

#     # ============================================================
#     # FORWARD (NEVER USED)
#     # ============================================================
#     def forward(self, x):
#         if self.hparams.flatten_temporal_dimension and x.ndim == 5:
#             x = x.flatten(1, 2)

#         with torch.no_grad():
#             f = self.Fs(x)[-1]

#         return self.D(f)


# import os
# from typing import Optional, Tuple, List

# import torch
# import torch.nn as nn
# import segmentation_models_pytorch as smp
# import h5py

# from torch.utils.data import DataLoader
# from itertools import cycle

# from ..BaseModel import BaseModel
# from dataloader.FireSpreadDataset import FireSpreadDataset


# # ============================================================
# # DOMAIN DISCRIMINATOR HEAD
# # ============================================================
# class DomainLogitHead(nn.Module):
#     """Small stable MLP: pooled feature map → logit."""
#     def __init__(self, in_channels: int):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Linear(in_channels, 512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.1),
#             nn.Linear(512, 128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.1),
#             nn.Linear(128, 1),
#         )

#     def forward(self, x):
#         return self.net(x).squeeze(1)


# # ============================================================
# # IWAN STAGE-2 MULTI-YEAR WEIGHT ESTIMATOR
# # ============================================================
# class IWANStage2_WeightEstimator(BaseModel):

#     def __init__(
#         self,
#         encoder_name: str = "resnet18",
#         encoder_weights: Optional[str] = None,
#         in_channels: Optional[int] = None,
#         n_channels: int = 7,
#         flatten_temporal_dimension: bool = True,
#         pos_class_weight: float = 1.0,
#         loss_function: str = "BCE",
#         use_doy: bool = False,
#         crop_before_eval: bool = False,
#         required_img_size: Optional[Tuple[int, int]] = None,
#         alpha_focal: Optional[float] = None,
#         f1_threshold: Optional[float] = None,
#         ckpt_dir: str = None,

#         # NEW ARGUMENTS
#         source_year: Optional[int] = None,
#         all_target_years: Optional[List[int]] = None,
#         save_dir: Optional[str] = None,

#         # training hyperparameters
#         inner_epochs: int = 3,
#         inner_steps_per_epoch: Optional[int] = None,
#         lr: float = 5e-5,

#         **kwargs,
#     ):

#         # allow override
#         if in_channels is not None:
#             n_channels = in_channels

#         super().__init__(
#             n_channels=n_channels,
#             flatten_temporal_dimension=flatten_temporal_dimension,
#             pos_class_weight=pos_class_weight,
#             loss_function=loss_function,
#             use_doy=use_doy,
#             crop_before_eval=crop_before_eval,
#             required_img_size=required_img_size,
#             alpha_focal=alpha_focal,
#             f1_threshold=f1_threshold,
#             **kwargs,
#         )

#         # save hparams
#         self.save_hyperparameters()

#         # ------------------------------
#         # ERRORS / REQUIRED ARGS
#         # ------------------------------
#         if ckpt_dir is None or not os.path.isdir(ckpt_dir):
#             raise FileNotFoundError(f"Stage-1 ckpt directory not found: {ckpt_dir}")

#         if source_year is None:
#             raise ValueError("source_year must be provided.")

#         if save_dir is None:
#             raise ValueError("save_dir must be provided by MyLightningCLI.")

#         self.source_year = int(source_year)
#         self.inner_epochs = int(inner_epochs)
#         self.inner_steps_per_epoch = inner_steps_per_epoch
#         self.lr = lr

#         # target list
#         if all_target_years is None:
#             self.all_target_years = list(range(2012, 2024))
#         else:
#             self.all_target_years = [int(y) for y in all_target_years]

#         # ------------------------------
#         # SET FINAL SAVE FILE
#         # ------------------------------
#         os.makedirs(save_dir, exist_ok=True)
#         self.weight_file = os.path.join(
#             save_dir, f"train_{self.source_year}_test_all.h5"
#         )

#         print(f"💾 IWAN weights will be saved to:\n   {self.weight_file}")

#         # ------------------------------
#         # LOAD STAGE-1 ENCODER (FROZEN)
#         # ------------------------------
#         ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
#         if not ckpt_files:
#             raise FileNotFoundError(f"No .ckpt file found in {ckpt_dir}")

#         ckpt_path = os.path.join(ckpt_dir, ckpt_files[0])
#         print(f"🔥 Loading Stage-1 encoder from: {ckpt_path}")

#         unet = smp.Unet(
#             encoder_name=encoder_name,
#             encoder_weights=None,
#             in_channels=n_channels,
#             classes=1,
#         )

#         ckpt = torch.load(ckpt_path, map_location="cpu")
#         sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

#         encoder_sd = {
#             k.replace("model.", ""): v
#             for k, v in sd.items()
#             if "encoder" in k
#         }

#         unet.load_state_dict(encoder_sd, strict=False)
#         self.Fs = unet.encoder

#         # freeze encoder fully
#         for p in self.Fs.parameters():
#             p.requires_grad_(False)

#         # disable BN
#         for m in self.Fs.modules():
#             if isinstance(m, nn.BatchNorm2d):
#                 m.track_running_stats = False
#                 m.running_mean = None
#                 m.running_var = None

#         # store feature size
#         self.feat_dim = self.Fs.out_channels[-1]

#         self.bce = nn.BCEWithLogitsLoss()

#     # ============================================================
#     # ENCODER APPLY
#     # ============================================================
#     @torch.no_grad()
#     def encode(self, x):
#         feats = self.Fs(x)
#         return feats[-1]

#     # ============================================================
#     # MAIN ENTRY POINT (CALLED FROM train.py)
#     # ============================================================
#     def run_full_iwan(self, datamodule):
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.to(device)
#         print(f"🟢 Using device: {device}")

#         # ---------------------------
#         # build source loader
#         # ---------------------------
#         datamodule.setup(stage="fit")
#         source_loader = datamodule.train_dataloader()
#         n_source = len(source_loader.dataset)

#         # prepare file
#         with h5py.File(self.weight_file, "a") as f:
#             if "sample_index" not in f:
#                 f.create_dataset("sample_index",
#                                  data=list(range(n_source)),
#                                  compression="gzip")

#         # ---------------------------
#         # LOOP OVER TARGET YEARS
#         # ---------------------------
#         for target_year in self.all_target_years:
#             print("\n" + "=" * 60)
#             print(f"🎯 TARGET YEAR = {target_year}")
#             print("=" * 60)

#             # build loader for this target year
#             target_loader = self._build_target_loader(datamodule, target_year)
#             if len(target_loader.dataset) == 0:
#                 print("⚠️ No samples. Skipping.")
#                 continue

#             # train discriminator
#             disc = DomainLogitHead(self.feat_dim).to(device)
#             opt = torch.optim.Adam(disc.parameters(), lr=self.lr)

#             self._train_one_pair(
#                 disc, opt, source_loader, target_loader, device
#             )

#             # save weights
#             self._save_weights_for_year(
#                 disc, source_loader, device, target_year
#             )

#         print("\n✅ Finished IWAN Stage-2")
#         print(f"📁 Saved all weights → {self.weight_file}")

#     # ============================================================
#     # BUILD TARGET LOADER
#     # ============================================================
#     def _build_target_loader(self, dm, year):
#         try:
#             norm_years = dm.train_dataset.included_fire_years
#         except:
#             norm_years = [self.source_year]

#         ds = FireSpreadDataset(
#             data_dir=dm.data_dir,
#             included_fire_years=[year],
#             n_leading_observations=dm.n_leading_observations,
#             n_leading_observations_test_adjustment=dm.n_leading_observations_test_adjustment,
#             crop_side_length=dm.crop_side_length,
#             load_from_hdf5=dm.load_from_hdf5,
#             is_train=False,
#             remove_duplicate_features=dm.remove_duplicate_features,
#             features_to_keep=dm.features_to_keep,
#             return_doy=dm.return_doy,
#             stats_years=norm_years,
#             is_pad=dm.is_pad,
#         )

#         return DataLoader(
#             ds, batch_size=dm.batch_size,
#             shuffle=True, num_workers=dm.num_workers,
#             pin_memory=True
#         )

#     # ============================================================
#     # TRAIN DISCRIMINATOR PER TARGET YEAR
#     # ============================================================
#     def _train_one_pair(self, disc, opt, source_loader, target_loader, device):
#         steps = min(len(source_loader), len(target_loader))
#         if self.inner_steps_per_epoch is not None:
#             steps = min(steps, self.inner_steps_per_epoch)

#         for ep in range(self.inner_epochs):
#             s_iter = cycle(source_loader)
#             t_iter = cycle(target_loader)

#             total_loss = 0
#             for _ in range(steps):
#                 x_s, _ = next(s_iter)
#                 x_t, _ = next(t_iter)

#                 x_s = x_s.to(device)
#                 x_t = x_t.to(device)

#                 if x_s.ndim == 5:
#                     x_s = x_s.flatten(1, 2)
#                 if x_t.ndim == 5:
#                     x_t = x_t.flatten(1, 2)

#                 f_s = self.encode(x_s)
#                 f_t = self.encode(x_t)

#                 log_s = disc(f_s)
#                 log_t = disc(f_t)

#                 y_s = torch.ones_like(log_s)
#                 y_t = torch.zeros_like(log_t)

#                 loss = 0.5 * self.bce(log_s, y_s) + \
#                        0.5 * self.bce(log_t, y_t)

#                 opt.zero_grad()
#                 loss.backward()
#                 opt.step()

#                 total_loss += loss.item()

#             print(f"   Epoch {ep+1}/{self.inner_epochs} – loss={total_loss/steps:.4f}")

#     # ============================================================
#     # SAVE WEIGHTS FOR THIS TARGET YEAR
#     # ============================================================
#     @torch.no_grad()
#     def _save_weights_for_year(self, disc, source_loader, device, year):
#         all_w = []

#         disc.eval()
#         for x_s, _ in source_loader:
#             x_s = x_s.to(device)
#             if x_s.ndim == 5:
#                 x_s = x_s.flatten(1, 2)

#             f_s = self.encode(x_s)
#             logits = disc(f_s)
#             p = torch.sigmoid(logits)

#             # IWAN weight = how target-like the sample is
#             w = p.cpu()
#             all_w.append(w)

#         weights = torch.cat(all_w, 0).numpy()

#         with h5py.File(self.weight_file, "a") as f:
#             ds_name = f"w_{year}"
#             if ds_name in f:
#                 del f[ds_name]
#             f.create_dataset(ds_name, data=weights, compression="gzip")

#         print(f"   💾 Saved weights → w_{year}")




import os
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import h5py
import wandb

from torch.utils.data import DataLoader
from itertools import cycle

from ..BaseModel import BaseModel
from dataloader.FireSpreadDataset import FireSpreadDataset


# ============================================================
# DOMAIN DISCRIMINATOR HEAD
# ============================================================
class DomainLogitHead(nn.Module):
    """Small stable MLP: pooled feature map → logit."""
    def __init__(self, in_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class DomainHead512(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)       


class DomainHead3x1024(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Linear(1024, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)



class DomainheadCNN(nn.Module):
    """
    Multi-layer CNN domain discriminator on spatial feature maps.
    Inspired by FCDiscriminator-style designs in UDA seg papers.
    Input:  feature map (B, C, H, W) from encoder
    Output: scalar logit per sample (B,)
    """
    def __init__(self, in_channels: int, base_channels: int = 256):
        super().__init__()

        c1 = base_channels
        c2 = base_channels // 2
        c3 = base_channels // 4

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(c3, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        # x: (B, C, H, W)
        logits_map = self.net(x)          # (B, 1, H, W)
        logits = logits_map.mean(dim=(2, 3))  # global average pool → (B,)
        return logits




# ============================================================
# IWAN STAGE-2 — GPU OPTIMIZED
# ============================================================
class IWANStage2_WeightEstimator(BaseModel):

    def __init__(
        self,
        encoder_name: str = "resnet18",
        encoder_weights: Optional[str] = None,
        in_channels: Optional[int] = None,
        n_channels: int = 7,
        flatten_temporal_dimension: bool = True,
        pos_class_weight: float = 1.0,
        loss_function: str = "BCE",
        use_doy: bool = False,
        crop_before_eval: bool = False,
        required_img_size: Optional[Tuple[int, int]] = None,
        alpha_focal: Optional[float] = None,
        f1_threshold: Optional[float] = None,
        ckpt_dir: str = None,

        # NEW ARGUMENTS
        source_year: Optional[int] = None,
        all_target_years: Optional[List[int]] = None,
        save_dir: Optional[str] = None,

        # training hyperparameters
        inner_epochs: int = 200,
        inner_steps_per_epoch: Optional[int] = None,
        lr: float = 5e-5,

        **kwargs,
    ):

        # allow override
        if in_channels is not None:
            n_channels = in_channels

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
            **kwargs,
        )

        self.save_hyperparameters()

        if ckpt_dir is None or not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(f"Stage-1 ckpt directory not found: {ckpt_dir}")

        if source_year is None:
            raise ValueError("source_year must be provided.")

        if save_dir is None:
            raise ValueError("save_dir must be provided.")

        self.source_year = int(source_year)
        self.inner_epochs = int(inner_epochs)
        self.inner_steps_per_epoch = inner_steps_per_epoch
        self.lr = lr

        # target list
        if all_target_years is None:
            self.all_target_years = list(range(2012, 2024))
        else:
            self.all_target_years = [int(y) for y in all_target_years]

        os.makedirs(save_dir, exist_ok=True)
        # Base path; final filenames will include target year
        self.weight_file_base = save_dir

        # ------------------------------
        # LOAD STAGE-1 ENCODER
        # ------------------------------
        ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
        if not ckpts:
            raise FileNotFoundError(f"No .ckpt file found in {ckpt_dir}")

        ckpt_path = os.path.join(ckpt_dir, ckpts[0])
        print(f"🔥 Loading Stage-1 encoder from: {ckpt_path}")

        unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=n_channels,
            classes=1,
        )

        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

        encoder_sd = {
            k.replace("model.", ""): v
            for k, v in sd.items()
            if "encoder" in k
        }

        unet.load_state_dict(encoder_sd, strict=False)
        self.Fs = unet.encoder

        for p in self.Fs.parameters():
            p.requires_grad = False

        # disable BN stats
        for m in self.Fs.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None

        self.feat_dim = self.Fs.out_channels[-1]
        self.bce = nn.BCEWithLogitsLoss()

        # Enable fast cuDNN
        torch.backends.cudnn.benchmark = True

    # ============================================================
    # ENCODER APPLY — OPTIMIZED
    # ============================================================
    # @torch.inference_mode()
    # def encode(self, x):
    #     return self.Fs(x)[-1]

    def encode(self, x):
    # encoder is frozen already but we should NOT use inference_mode
        with torch.no_grad():
            f = self.Fs(x)[-1]
        return f.clone()   # gives autograd-friendly tensor for CNN

    # ============================================================
    # MAIN ENTRY — GPU OPTIMIZED
    # ============================================================
    def run_full_iwan(self, datamodule):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        print(f"🟢 Using device: {device}")

        datamodule.setup(stage="fit")
        source_loader = datamodule.train_dataloader()
        n_source = len(source_loader.dataset)

        # We will write one HDF5 per target year; no preallocation here

        # ---------------------------
        # TARGET-YEAR LOOP
        # ---------------------------
        for target_year in self.all_target_years:
            print("\n" + "=" * 60)
            print(f"🎯 TARGET YEAR = {target_year}")
            print("=" * 60)

            target_loader = self._build_target_loader(datamodule, target_year)
            if len(target_loader.dataset) == 0:
                print("⚠️ No data, skipping.")
                continue

            # simple source/target train-val split AFTER labeling
            src_train_loader, src_val_loader = self._split_loader(
                source_loader, datamodule.batch_size, datamodule.num_workers, datamodule, is_source=True
            )
            tgt_train_loader, tgt_val_loader = self._split_loader(
                target_loader, datamodule.batch_size, datamodule.num_workers, datamodule, is_source=False
            )

            history = []
            disc = DomainheadCNN(self.feat_dim).to(device)
            opt = torch.optim.Adam(disc.parameters(), lr=self.lr)
            scaler = torch.cuda.amp.GradScaler()

            self._train_one_pair(
                disc, opt, scaler,
                src_train_loader, tgt_train_loader,
                src_val_loader, tgt_val_loader,
                device, target_year, history
            )

            self._save_weights_for_year(
                disc, source_loader, device, target_year, history
            )

        print("\n✅ Finished IWAN Stage-2")
        print(f"📁 Saved all weights under: {self.weight_file_base}")

    # ============================================================
    # TARGET LOADER
    # ============================================================
    def _build_target_loader(self, dm, year):
        try:
            norm_years = dm.train_dataset.included_fire_years
        except:
            norm_years = [self.source_year]

        ds = FireSpreadDataset(
            data_dir=dm.data_dir,
            included_fire_years=[year],
            n_leading_observations=dm.n_leading_observations,
            n_leading_observations_test_adjustment=dm.n_leading_observations_test_adjustment,
            crop_side_length=dm.crop_side_length,
            load_from_hdf5=dm.load_from_hdf5,
            is_train=True,
            remove_duplicate_features=dm.remove_duplicate_features,
            features_to_keep=dm.features_to_keep,
            return_doy=dm.return_doy,
            stats_years=norm_years,
            is_pad=dm.is_pad,
        )

        return DataLoader(
            ds,
            batch_size=dm.batch_size,
            shuffle=True,
            num_workers=dm.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    # ============================================================
    # TRAIN ONE PAIR — AMP + GPU FAST
    # ============================================================
    def _train_one_pair(self, disc, opt, scaler,
                        src_loader, tgt_loader, src_val_loader, tgt_val_loader, device, target_year, history):

        steps = min(len(src_loader), len(tgt_loader))
        if self.inner_steps_per_epoch:
            steps = min(steps, self.inner_steps_per_epoch)

        s_iter = cycle(src_loader)
        t_iter = cycle(tgt_loader)

        for ep in range(self.inner_epochs):
            total_loss = 0
            correct_s = correct_t = 0
            total_s = total_t = 0

            for _ in range(steps):
                x_s, _ = next(s_iter)
                x_t, _ = next(t_iter)

                x_s = x_s.to(device, non_blocking=True)
                x_t = x_t.to(device, non_blocking=True)

                if x_s.ndim == 5:
                    x_s = x_s.flatten(1, 2)
                if x_t.ndim == 5:
                    x_t = x_t.flatten(1, 2)

                with torch.cuda.amp.autocast():
                    f_s = self.encode(x_s)
                    f_t = self.encode(x_t)

                    log_s = disc(f_s)
                    log_t = disc(f_t)

                    y_s = torch.ones_like(log_s, device=device)
                    y_t = torch.zeros_like(log_t, device=device)

                    loss = 0.5 * self.bce(log_s, y_s) + \
                           0.5 * self.bce(log_t, y_t)

                    p_s = (torch.sigmoid(log_s) > 0.5).float()
                    p_t = (torch.sigmoid(log_t) < 0.5).float()

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                total_loss += loss.item()
                correct_s += p_s.sum().item()
                correct_t += p_t.sum().item()
                total_s += p_s.numel()
                total_t += p_t.numel()

            train_loss = total_loss / steps
            val_loss_value, val_acc = self._eval_pair(disc, src_val_loader, tgt_val_loader, device)

            train_acc = 0.5 * ((correct_s / max(1, total_s)) + (correct_t / max(1, total_t)))

            history.append({
                "epoch": ep + 1,
                "train_loss": train_loss,
                "val_loss": val_loss_value,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "target_year": target_year,
            })

            # log to wandb if available
            try:
                wandb.log({
                    "train_loss_D": train_loss,
                    "val_loss_D": val_loss_value,
                    "train_acc_D": train_acc,
                    "val_acc_D": val_acc,
                    "target_year": target_year,
                    "epoch": ep + 1,
                })
            except Exception:
                pass

            print(f"   Epoch {ep+1}/{self.inner_epochs} – train_loss={train_loss:.4f} val_loss={val_loss_value:.4f} train_acc={train_acc:.4f} val_acc={(val_acc if val_acc is not None else float('nan')):.4f}")

    # ============================================================
    # SAVE IWAN WEIGHTS — GPU FAST
    # ============================================================
    @torch.inference_mode()
    def _save_weights_for_year(self, disc, source_loader, device, year, history):
        disc.eval()
        out = []

        for x_s, _ in source_loader:
            x_s = x_s.to(device, non_blocking=True)
            if x_s.ndim == 5:
                x_s = x_s.flatten(1, 2)

            f_s = self.encode(x_s)
            # logits = disc(f_s)
            # p = torch.sigmoid(logits)
            # out.append(p.cpu())

            logits = disc(f_s)
            D_star = torch.sigmoid(logits)   # probability of source
            w = 1.0 - D_star 
            out.append(w.cpu())                # IWAN importance weight

        weights = torch.cat(out, 0).numpy()

        # Save to per-target-year file
        weight_file = os.path.join(
            self.weight_file_base, f"single_layer_cnnmodel_allothertrain_test_{year}.h5"
        )
        with h5py.File(weight_file, "a") as f:
            if "sample_index" not in f:
                f.create_dataset("sample_index",
                                 data=list(range(len(source_loader.dataset))),
                                 compression="gzip")
            f.create_dataset("w", data=weights, compression="gzip")

        print(f"   💾 Saved weights → {weight_file}")


        # --------------------------------------------------------
        # NEW: SAVE CHECKPOINT TO LIGHTNING DEFAULT CHECKPOINT DIR
        # --------------------------------------------------------
        # Save stage-2 checkpoints inside save_dir/stage2_checkpoints
        ckpt_dir = os.path.join(self.weight_file_base, "stage2_checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        ckpt_path = os.path.join(
            ckpt_dir, f"iwan_stage2_CNNmodel_target_year{year}.ckpt"
        )

        torch.save({
            "discriminator": disc.state_dict(),
            "feat_dim": self.feat_dim,
            "target_year": year,
            "source_year": self.source_year,
            "history": history,
        }, ckpt_path)

        print(f"   📌 Saved Stage-2 checkpoint → {ckpt_path}")

    @torch.inference_mode()
    def _eval_pair(self, disc, src_loader, tgt_loader, device):
        total_loss = 0.0
        total_count = 0
        correct_s = correct_t = 0
        total_s = total_t = 0
        # source: label 1
        for x_s, _ in src_loader:
            x_s = x_s.to(device, non_blocking=True)
            if x_s.ndim == 5:
                x_s = x_s.flatten(1, 2)
            f_s = self.encode(x_s)
            log_s = disc(f_s)
            y_s = torch.ones_like(log_s, device=device)
            loss = 0.5 * self.bce(log_s, y_s)
            total_loss += loss.item() * x_s.size(0)
            total_count += x_s.size(0)
            p_s = (torch.sigmoid(log_s) > 0.5).float()
            correct_s += p_s.sum().item()
            total_s += p_s.numel()
        # target: label 0
        for x_t, _ in tgt_loader:
            x_t = x_t.to(device, non_blocking=True)
            if x_t.ndim == 5:
                x_t = x_t.flatten(1, 2)
            f_t = self.encode(x_t)
            log_t = disc(f_t)
            y_t = torch.zeros_like(log_t, device=device)
            loss = 0.5 * self.bce(log_t, y_t)
            total_loss += loss.item() * x_t.size(0)
            total_count += x_t.size(0)
            p_t = (torch.sigmoid(log_t) < 0.5).float()
            correct_t += p_t.sum().item()
            total_t += p_t.numel()

        val_loss = total_loss / max(1, total_count)
        val_acc = 0.5 * ((correct_s / max(1, total_s)) + (correct_t / max(1, total_t)))
        return val_loss, val_acc

    # --------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------
    def _split_loader(self, loader, batch_size, num_workers, datamodule, is_source: bool):
        # Take a small validation split from an existing loader’s dataset
        ds = loader.dataset
        total_len = len(ds)
        val_len = max(1, int(0.1 * total_len))
        train_len = max(1, total_len - val_len)
        gen = torch.Generator().manual_seed(42)
        train_ds, val_ds = torch.utils.data.random_split(ds, [train_len, val_len], generator=gen)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, persistent_workers=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True, persistent_workers=True
        )
        return train_loader, val_loader
