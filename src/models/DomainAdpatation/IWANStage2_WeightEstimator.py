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
    @torch.inference_mode()
    def encode(self, x):
        return self.Fs(x)[-1]

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

            disc = DomainHead512(self.feat_dim).to(device)
            opt = torch.optim.Adam(disc.parameters(), lr=self.lr)
            scaler = torch.cuda.amp.GradScaler()

            self._train_one_pair(
                disc, opt, scaler,
                source_loader, target_loader, device
            )

            self._save_weights_for_year(
                disc, source_loader, device, target_year
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
                        source_loader, target_loader, device):

        steps = min(len(source_loader), len(target_loader))
        if self.inner_steps_per_epoch:
            steps = min(steps, self.inner_steps_per_epoch)

        s_iter = cycle(source_loader)
        t_iter = cycle(target_loader)

        for ep in range(self.inner_epochs):
            total_loss = 0

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

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                total_loss += loss.item()

            print(f"   Epoch {ep+1}/{self.inner_epochs} – loss={total_loss/steps:.4f}")

    # ============================================================
    # SAVE IWAN WEIGHTS — GPU FAST
    # ============================================================
    @torch.inference_mode()
    def _save_weights_for_year(self, disc, source_loader, device, year):
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
            self.weight_file_base, f"512_single_layerallothertrain_test_{year}.h5"
        )
        with h5py.File(weight_file, "a") as f:
            if "sample_index" not in f:
                f.create_dataset("sample_index",
                                 data=list(range(len(source_loader.dataset))),
                                 compression="gzip")
            f.create_dataset("w", data=weights, compression="gzip")

        print(f"   💾 Saved weights → {weight_file}")
