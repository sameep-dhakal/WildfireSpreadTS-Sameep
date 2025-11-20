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
# DOMAIN HEAD (DISCRIMINATOR)
# ============================================================
class DomainLogitHead(nn.Module):
    """
    Simple domain discriminator:
        input: feature map (B, C, H, W) from encoder
        output: logit (B,)
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # (B, C, H, W) -> (B, C, 1, 1)
            nn.Flatten(),              # (B, C)
            nn.Linear(in_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 1),         # logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B,)
        return self.net(x).squeeze(1)


# ============================================================
# IWAN STAGE 2 – MULTI-YEAR WEIGHT ESTIMATOR
# ============================================================
class IWANStage2_WeightEstimator(BaseModel):
    """
    IWAN Stage-2 (self-contained version):

    - Uses a frozen encoder Fs from Stage-1 UNet.
    - For ONE fixed source year (e.g. 2012), loops over ALL target years:
        target_year in [2012, 2013, ..., 2023]

      For each target_year:
        1) Build target DataLoader for that year only.
        2) Train a domain discriminator D_target on (source vs that target).
           Labels: source = 1, target = 0.
        3) Compute IWAN weights for ALL source samples:
               w_i^(target_year) = p(target | x_i)
        4) Save weights into a single HDF5 file:

               <trainer.default_root_dir>/train_{source_year}_test_all.h5

           datasets:
               /sample_index          (0 .. N_source-1) (written once)
               /w_2012, /w_2013, ...  (float32 arrays of size N_source)
    """

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
        # inner training hyper-params (per target year)
        inner_epochs: int = 3,
        inner_steps_per_epoch: Optional[int] = None,  # None -> full epoch
        lr: float = 5e-5,
        **kwargs,
    ):
        # Allow overriding n_channels
        if in_channels is not None:
            n_channels = in_channels

        # Initialize BaseModel just to keep consistency with your codebase
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

        # Save all hparams (Lightning-compatible)
        self.save_hyperparameters()

        # ------------------------------
        # SANITY CHECKS
        # ------------------------------
        if ckpt_dir is None or (not os.path.isdir(ckpt_dir)):
            raise FileNotFoundError(f"Stage-1 ckpt directory not found: {ckpt_dir}")

        if source_year is None:
            raise ValueError("source_year must be provided to IWANStage2_WeightEstimator.")

        self.source_year = int(source_year)
        self.inner_epochs = int(inner_epochs)
        self.inner_steps_per_epoch = inner_steps_per_epoch
        self.lr = lr

        # If target list not provided, hard-code all years
        if all_target_years is None:
            self.all_target_years = [
                2012, 2013, 2014, 2015,
                2016, 2017, 2018, 2019,
                2020, 2021, 2022, 2023
            ]
        else:
            self.all_target_years = [int(y) for y in all_target_years]

        # NOTE: we will set self.weight_file later in run_full_iwan(),
        # using trainer.default_root_dir. Do NOT build it here.

        # ------------------------------
        # LOAD STAGE-1 ENCODER
        # ------------------------------
        ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
        if not ckpt_files:
            raise FileNotFoundError(f"No Stage-1 .ckpt found in {ckpt_dir}")

        ckpt_path = os.path.join(ckpt_dir, ckpt_files[0])
        print(f"✅ Loading Stage-1 encoder from: {ckpt_path}")

        unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=n_channels,
            classes=1,
        )

        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)

        encoder_state = {
            k.replace("model.", ""): v
            for k, v in state_dict.items()
            if "encoder" in k
        }

        unet.load_state_dict(encoder_state, strict=False)
        self.Fs = unet.encoder  # frozen feature extractor

        # Freeze encoder
        for p in self.Fs.parameters():
            p.requires_grad_(False)

        # Disable BN running stats
        for m in self.Fs.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None

        # ------------------------------
        # Domain discriminator template
        # (we'll re-create / re-init per target year)
        # ------------------------------
        feat_dim = self.Fs.out_channels[-1]
        self.disc_template_in_channels = feat_dim

        # Just a placeholder so BaseModel has some module
        self.bce = nn.BCEWithLogitsLoss()

    # ------------------------------------------------------------------
    # SIMPLE ENCODER WRAPPER
    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) OR (B, T*C, H, W) if flattened temporal.
        Returns final encoder feature map: (B, C_feat, H_feat, W_feat)
        """
        with torch.no_grad():
            feats = self.Fs(x)
        return feats[-1]

    # ------------------------------------------------------------------
    # MANUAL TRAINING ENTRY POINT
    # ------------------------------------------------------------------
    def run_full_iwan(self, datamodule) -> None:
        """
        Main entry point for Stage-2:

        Call this from train.py instead of trainer.fit()

            model.run_full_iwan(datamodule)

        It will:
          - build source loader (train_years = [source_year])
          - loop over all target years
          - train a fresh discriminator for each target year
          - compute weights
          - save everything into HDF5
        """
        # ------------------ DEVICE SETUP ------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        print(f"🟢 Using device: {device}")

        # ------------------ DECIDE SAVE DIRECTORY ------------------
        # Prefer trainer.default_root_dir (e.g. /develop/results/domain_adaptation_stage2_outputs/fold0)
        if hasattr(self, "trainer") and getattr(self.trainer, "default_root_dir", None):
            base_dir = self.trainer.default_root_dir
        else:
            # fallback: current working directory
            base_dir = os.getcwd()

        os.makedirs(base_dir, exist_ok=True)
        self.weight_file = os.path.join(
            base_dir, f"train_{self.source_year}_test_all.h5"
        )
        print(f"📁 IWAN weights will be saved to:\n   {self.weight_file}")

        # ------------------ PREPARE SOURCE LOADER ------------------
        datamodule.setup(stage="fit")

        source_loader: DataLoader = datamodule.train_dataloader()
        n_source_samples = len(source_loader.dataset)
        print(f"🔹 Source year = {self.source_year} with {n_source_samples} samples")

        # ------------------ HDF5 FILE PREP ------------------
        with h5py.File(self.weight_file, "a") as f:
            if "sample_index" not in f:
                f.create_dataset(
                    "sample_index",
                    data=list(range(n_source_samples)),
                    compression="gzip"
                )
                print("✅ Created dataset: sample_index")
            else:
                print("ℹ️ sample_index already exists, will reuse.")

        # ------------------ LOOP OVER TARGET YEARS ------------------
        for target_year in self.all_target_years:
            print("\n" + "=" * 70)
            print(f"🎯 TARGET YEAR: {target_year}")
            print("=" * 70)

            # 1) Build target DataLoader for this specific year
            target_loader = self._build_target_loader_for_year(datamodule, target_year)
            n_target_samples = len(target_loader.dataset)
            print(f"  - target year {target_year} has {n_target_samples} samples")

            if n_target_samples == 0:
                print(f"  ⚠️ Skipping year {target_year}: no samples found.")
                continue

            # 2) Train discriminator for (source_year vs target_year)
            disc = DomainLogitHead(self.disc_template_in_channels).to(device)
            optimizer = torch.optim.Adam(disc.parameters(), lr=self.lr)

            self._train_discriminator_for_pair(
                disc=disc,
                optimizer=optimizer,
                source_loader=source_loader,
                target_loader=target_loader,
                device=device,
                target_year=target_year,
            )

            # 3) Compute and save IWAN weights for ALL source samples
            self._compute_and_save_weights_for_year(
                disc=disc,
                source_loader=source_loader,
                device=device,
                target_year=target_year,
            )

        print("\n✅ IWAN Stage-2 completed for source year:", self.source_year)
        print(f"   All weights stored in: {self.weight_file}")

    # ------------------------------------------------------------------
    # BUILD TARGET LOADER FOR A GIVEN YEAR
    # ------------------------------------------------------------------
    def _build_target_loader_for_year(self, dm, target_year: int) -> DataLoader:
        """
        Build a fresh FireSpreadDataset / DataLoader for a single target year.
        We reuse datamodule's basic config (data_dir, n_leading_observations, etc.).
        """
        try:
            train_years = dm.train_dataset.included_fire_years
        except AttributeError:
            train_years = [self.source_year]

        target_dataset = FireSpreadDataset(
            data_dir=dm.data_dir,
            included_fire_years=[target_year],
            n_leading_observations=dm.n_leading_observations,
            n_leading_observations_test_adjustment=dm.n_leading_observations_test_adjustment,
            crop_side_length=dm.crop_side_length,
            load_from_hdf5=dm.load_from_hdf5,
            is_train=False,
            remove_duplicate_features=dm.remove_duplicate_features,
            features_to_keep=dm.features_to_keep,
            return_doy=dm.return_doy,
            stats_years=train_years,
            is_pad=dm.is_pad,
        )

        target_loader = DataLoader(
            target_dataset,
            batch_size=dm.batch_size,
            shuffle=True,
            num_workers=dm.num_workers,
            pin_memory=True,
        )

        return target_loader

    # ------------------------------------------------------------------
    # TRAIN DISCRIMINATOR FOR ONE (SOURCE, TARGET_YEAR) PAIR
    # ------------------------------------------------------------------
    def _train_discriminator_for_pair(
        self,
        disc: nn.Module,
        optimizer: torch.optim.Optimizer,
        source_loader: DataLoader,
        target_loader: DataLoader,
        device: torch.device,
        target_year: int,
    ) -> None:

        disc.train()
        bce = nn.BCEWithLogitsLoss()

        steps_per_epoch = min(len(source_loader), len(target_loader))
        if self.inner_steps_per_epoch is not None:
            steps_per_epoch = min(steps_per_epoch, self.inner_steps_per_epoch)

        print(f"  🔁 Training discriminator for year {target_year}: "
              f"{self.inner_epochs} epochs, {steps_per_epoch} steps/epoch")

        for epoch in range(self.inner_epochs):
            running_loss = 0.0

            source_iter = cycle(source_loader)
            target_iter = cycle(target_loader)

            for _ in range(steps_per_epoch):
                x_s, _ = next(source_iter)
                x_t, _ = next(target_iter)

                x_s = x_s.to(device)
                x_t = x_t.to(device)

                if self.hparams.flatten_temporal_dimension and x_s.ndim == 5:
                    x_s = x_s.flatten(1, 2)
                if self.hparams.flatten_temporal_dimension and x_t.ndim == 5:
                    x_t = x_t.flatten(1, 2)

                f_s = self.encode(x_s)
                f_t = self.encode(x_t)

                log_s = disc(f_s)
                log_t = disc(f_t)

                y_s = torch.ones_like(log_s)
                y_t = torch.zeros_like(log_t)

                loss = 0.5 * bce(log_s, y_s) + 0.5 * bce(log_t, y_t)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / steps_per_epoch
            print(f"    Epoch {epoch+1}/{self.inner_epochs} "
                  f"– target {target_year}: loss = {avg_loss:.4f}")

    # ------------------------------------------------------------------
    # COMPUTE & SAVE IWAN WEIGHTS FOR ONE TARGET YEAR
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _compute_and_save_weights_for_year(
        self,
        disc: nn.Module,
        source_loader: DataLoader,
        device: torch.device,
        target_year: int,
    ) -> None:

        disc.eval()
        all_w = []

        for x_s, _ in source_loader:
            x_s = x_s.to(device)

            if self.hparams.flatten_temporal_dimension and x_s.ndim == 5:
                x_s = x_s.flatten(1, 2)

            f_s = self.encode(x_s)
            log_s = disc(f_s)
            p_source = torch.sigmoid(log_s)

            # IWAN weight = probability of being "target-like"
            w = p_source  # shape: (B,)

            all_w.append(w.cpu())

        weights = torch.cat(all_w, dim=0).numpy()
        print(f"  💾 Saving {len(weights)} weights for target_year={target_year}")

        with h5py.File(self.weight_file, "a") as f:
            ds_name = f"w_{target_year}"
            if ds_name in f:
                del f[ds_name]
                print(f"    ℹ️ Overwriting existing dataset: {ds_name}")
            f.create_dataset(ds_name, data=weights, compression="gzip")

        print(f"  ✅ Saved weights as dataset: w_{target_year}")
