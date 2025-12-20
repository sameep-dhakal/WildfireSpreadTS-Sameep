# import os
# import torch
# import numpy as np
# import h5py
# from tqdm import tqdm
# from torch.utils.data import DataLoader

# from models.SMPModel import SMPModel
# from dataloader.FireSpreadDataset import FireSpreadDataset
# from collections import Counter
# from models.DomainAdpatation.IWANStage2_WeightEstimator import DomainHead3x1024

# # Match the Stage-2 data pipeline defaults (from the sweep YAML)
# STAGE2_FEATURES_TO_KEEP = [0, 1, 2, 3, 4, 38, 39]
# STAGE2_REMOVE_DUPLICATE_FEATURES = True
# STAGE2_RETURN_DOY = False
# STAGE2_N_LEADING_OBS = 1
# STAGE2_CROP_SIDE_LENGTH = 128
# STAGE2_LOAD_FROM_HDF5 = True
# STAGE2_IS_PAD = False

# # =====================================================================
# # Inspect checkpoints
# # =====================================================================
# def inspect_checkpoint(ckpt_path):
#     print("\n========================================")
#     print(f"🔍 Inspecting checkpoint:\n{ckpt_path}")
#     print("========================================")

#     ckpt = torch.load(ckpt_path, map_location="cpu")

#     if "discriminator" not in ckpt:
#         print("❌ No 'discriminator' key in checkpoint!")
#         print("Available keys:", ckpt.keys())
#         return

#     disc_state = ckpt["discriminator"]

#     print("\n📌 Discriminator State Dict Keys:")
#     for k, v in disc_state.items():
#         print(f"  {k:40s} {tuple(v.shape)}")

#     print("\n🎯 Feature Dim:", ckpt.get("feat_dim", "NOT FOUND"))
#     print("🎯 Target Year:", ckpt.get("target_year", "NOT FOUND"))

#     print("========================================\n")


# # =====================================================================
# # Extract true source years without relying on per-sample lookups
# # =====================================================================
# def get_source_years(dataset):
#     # dataset.datapoints_per_fire maps year -> {fire_name: count}
#     year_counts = []
#     for year, fires in dataset.datapoints_per_fire.items():
#         cnt = sum(fires.values())
#         if cnt > 0:
#             year_counts.append((int(year), cnt))

#     total = sum(c for _, c in year_counts)
#     years = np.empty(total, dtype=int)
#     idx = 0
#     for year, cnt in year_counts:
#         years[idx:idx+cnt] = year
#         idx += cnt

#     return years


# def summarize_year_counts(dataset):
#     """Return a dict of {year: num_samples} based on dataset internals."""
#     counts = {}
#     for fire_year, fires in dataset.datapoints_per_fire.items():
#         counts[int(fire_year)] = sum(fires.values())
#     return counts


# # =====================================================================
# # EXPORT FUNCTION — FIXED + T=1 HANDLING
# # =====================================================================
# def export_weights_for_fold(
#     fold,
#     stage1_ckpt_dir,
#     stage2_ckpt_path,
#     data_dir,
#     save_dir,
#     batch_size=64,
#     num_workers=8,
#     device=None,
#     features_to_keep=STAGE2_FEATURES_TO_KEEP,
# ):
#     # Auto-select device if not provided
#     if device is None:
#         device = "cuda" if torch.cuda.is_available() else "cpu"

#     print("\n==============================")
#     print(f"🔥 Fold {fold}")
#     print(f"Stage-1 ckpt dir: {stage1_ckpt_dir}")
#     print(f"Stage-2 checkpoint: {stage2_ckpt_path}")
#     print(f"Device: {device}")
#     print("==============================\n")

#     # ---------------------------------------------------
#     # Load Stage-1 encoder
#     # ---------------------------------------------------
#     ckpts = [f for f in os.listdir(stage1_ckpt_dir) if f.endswith(".ckpt")]
#     if not ckpts:
#         raise RuntimeError(f"No checkpoint found in {stage1_ckpt_dir}")

#     stage1_ckpt = os.path.join(stage1_ckpt_dir, ckpts[0])
#     print(f"Using Stage-1 ckpt: {stage1_ckpt}")

#     base = SMPModel.load_from_checkpoint(stage1_ckpt)
#     encoder = base.model.encoder
#     encoder.to(device)
#     encoder.eval()

#     expected_channels = base.hparams.n_channels
#     print(f"Stage-1 expected channels = {expected_channels}")

#     # ---------------------------------------------------
#     # Load Stage-2 checkpoint (discriminator)
#     # ---------------------------------------------------
#     ckpt2 = torch.load(stage2_ckpt_path, map_location=device)
#     feat_dim = ckpt2["feat_dim"]
#     target_year = ckpt2["target_year"]
#     source_years = ckpt2.get("source_years")
#     if source_years is None:
#         src_year = ckpt2.get("source_year")
#         source_years = [int(src_year)] if src_year is not None else list(range(2012, 2024))
#     source_years = [int(y) for y in source_years]
#     if len(source_years) == 1:
#         # The Stage-2 ckpt only recorded a single source_year; fall back to full range
#         print("⚠️ Stage-2 ckpt lists a single source year; using full 2012–2023 range to recover per-sample years.")
#         source_years = list(range(2012, 2024))

#     print(f"Target year for fold {fold}: {target_year}")
#     print(f"Source years from Stage-2 ckpt: {source_years}")

#     disc = DomainHead3x1024(feat_dim).to(device)
#     disc.load_state_dict(ckpt2["discriminator"])
#     disc.eval()

#     # ---------------------------------------------------
#     # Dataset built EXACTLY like Stage-2 training used it
#     # ---------------------------------------------------
#     source_dataset = FireSpreadDataset(
#         data_dir=data_dir,
#         included_fire_years=source_years,
#         n_leading_observations=STAGE2_N_LEADING_OBS,
#         n_leading_observations_test_adjustment=None,
#         crop_side_length=STAGE2_CROP_SIDE_LENGTH,
#         load_from_hdf5=STAGE2_LOAD_FROM_HDF5,
#         is_train=True,  # Stage-2 trained on the train split with augmentations
#         remove_duplicate_features=STAGE2_REMOVE_DUPLICATE_FEATURES,
#         stats_years=source_years,
#         features_to_keep=features_to_keep,
#         return_doy=STAGE2_RETURN_DOY,
#         is_pad=STAGE2_IS_PAD,
#     )

#     # Quick per-year summary to catch missing data early
#     year_counts = summarize_year_counts(source_dataset)
#     print("Per-year sample counts (source_dataset):", year_counts)

#     test_x, _ = source_dataset[0]

#     # Collapse temporal dimension when T=1 to match encoder expectations
#     if test_x.ndim == 4 and test_x.shape[0] == 1:      # [T, C, H, W]
#         test_x = test_x[0]                             # -> [C, H, W]
#     elif test_x.ndim == 4 and test_x.shape[0] != 1:
#         raise RuntimeError(f"Expected T=1 but got T={test_x.shape[0]}")

#     print(f"Dataset sample shape AFTER preprocessing: {test_x.shape}")

#     if test_x.shape[0] != expected_channels:
#         raise RuntimeError(
#             f"Dataset produces {test_x.shape[0]} channels but Stage-1 expects {expected_channels}"
#         )

#     src_loader = DataLoader(
#         source_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True,
#     )

#     # ---------------------------------------------------
#     # True source-year array
#     # ---------------------------------------------------
#     print("Extracting src_year array…")
#     src_years = get_source_years(source_dataset)
#     year_freq = Counter(src_years)
#     print(f"Unique src_years found: {sorted(year_freq.items())}")

#     # ---------------------------------------------------
#     # Compute IWAN weights
#     # ---------------------------------------------------
#     print("Computing importance weights…")
#     weights = []

#     with torch.no_grad():
#         for x, _ in tqdm(src_loader):

#             # 🟢 FIX: Dataset returns 5D tensor [B, T, C, H, W] when T=1 → collapse
#             if x.ndim == 5:
#                 if x.shape[1] != 1:
#                     raise RuntimeError(f"Expected T=1 but got T={x.shape[1]}")
#                 x = x[:, 0]  # remove temporal dimension → [B, C, H, W]
#             elif x.ndim == 4 and test_x.ndim == 3:
#                 # Already collapsed per-sample; nothing to do
#                 pass
#             elif x.ndim == 4:
#                 # Handle [B, T, C, H, W] already flattened by dataset? ensure T=1 case
#                 if x.shape[1] == expected_channels:
#                     # shape [B, C, H, W]
#                     pass
#                 else:
#                     raise RuntimeError(f"Unexpected input shape {tuple(x.shape)}")

#             x = x.to(device)

#             feats = encoder(x)[-1]
#             logits = disc(feats)

#             D = torch.sigmoid(logits)    # probability being SOURCE
#             w = 1.0 - D                  # IWAN importance weight

#             weights.append(w.cpu().numpy())

#     weights = np.concatenate(weights, axis=0)

#     # ---------------------------------------------------
#     # Save output
#     # ---------------------------------------------------
#     os.makedirs(save_dir, exist_ok=True)
#     out_path = os.path.join(save_dir, f"weights_new_srcyears_target_{target_year}.h5")

#     print("Saving:", out_path)
#     with h5py.File(out_path, "w") as f:
#         f.create_dataset("sample_index", data=np.arange(len(source_dataset)))
#         f.create_dataset("w", data=weights)
#         f.create_dataset("src_year", data=src_years)
#         f.attrs["source_years"] = np.array(source_years, dtype=int)
#         f.attrs["target_year"] = int(target_year)

#     print(f"✅ Fold {fold} completed\n")


# # =====================================================================
# # MAIN
# # =====================================================================
# if __name__ == "__main__":
#     print("🚀 Starting Stage-2 export…")

#     STAGE1_CKPT_DIR = {
#         1: "/develop/results/wildfire-progression/hvsbcl8a/checkpoints",
#         2: "/develop/results/wildfire-progression/zsqvrj9f/checkpoints",
#         3: "/develop/results/wildfire-progression/04zyztgf/checkpoints",
#         4: "/develop/results/wildfire-progression/kf5e2z7i/checkpoints",
#         5: "/develop/results/wildfire-progression/z4fpm67c/checkpoints",
#         6: "/develop/results/wildfire-progression/xu8nf4pr/checkpoints",
#         7: "/develop/results/wildfire-progression/3gx6vn1b/checkpoints",
#         8: "/develop/results/wildfire-progression/gbco149i/checkpoints",
#         9: "/develop/results/wildfire-progression/u2jopt8y/checkpoints",
#         10: "/develop/results/wildfire-progression/asvp9e1m/checkpoints",
#         11: "/develop/results/wildfire-progression/6l528lvo/checkpoints",
#         12: "/develop/results/wildfire-progression/o7t67rsw/checkpoints",
#     }

#     STAGE2_CKPT_DIR = {
#         1: "/develop/results/stage2_checkpoints/iwan_stage2_1024_new_allothertrain_target_year2013.ckpt",
#         2: "/develop/results/stage2_checkpoints/iwan_stage2_1024_new_allothertrain_target_year2014.ckpt",
#         3: "/develop/results/stage2_checkpoints/iwan_stage2_1024_new_allothertrain_target_year2015.ckpt",
#         4: "/develop/results/stage2_checkpoints/iwan_stage2_1024_new_allothertrain_target_year2016.ckpt",
#         5: "/develop/results/stage2_checkpoints/iwan_stage2_1024_new_allothertrain_target_year2017.ckpt",
#         6: "/develop/results/stage2_checkpoints/iwan_stage2_1024_new_allothertrain_target_year2018.ckpt",
#         7: "/develop/results/stage2_checkpoints/iwan_stage2_1024_new_allothertrain_target_year2019.ckpt",
#         8: "/develop/results/stage2_checkpoints/iwan_stage2_1024_new_allothertrain_target_year2020.ckpt",
#         9: "/develop/results/stage2_checkpoints/iwan_stage2_1024_new_allothertrain_target_year2021.ckpt",
#         10: "/develop/results/stage2_checkpoints/iwan_stage2_1024_new_allothertrain_target_year2022.ckpt",
#         11: "/develop/results/stage2_checkpoints/iwan_stage2_1024_new_allothertrain_target_year2023.ckpt",
#     }

#     DATA_DIR = "/develop/data/WildfireSpreadTS_2012_2015_hdf5/"
#     SAVE_DIR = "/develop/results/stage2_weight_exports/"

#     for fold in STAGE1_CKPT_DIR:
#         export_weights_for_fold(
#             fold=fold,
#             stage1_ckpt_dir=STAGE1_CKPT_DIR[fold],
#             stage2_ckpt_path=STAGE2_CKPT_DIR[fold],
#             data_dir=DATA_DIR,
#             save_dir=SAVE_DIR,
#         )

#     print("\n🎉 ALL FOLDS COMPLETED SUCCESSFULLY!")



import os
import torch
from pytorch_lightning import Trainer
from models.SMPModel import SMPModel
from dataloader.FireSpreadDataModule import FireSpreadDataModule

def diagnostic_test_stage1(stage1_ckpt_path, target_year, data_dir):
    """
    Diagnose Stage 1 Expert performance on the Target Year only.
    This helps determine if the data for that year is fundamentally difficult.
    """
    print(f"\n🚀 Starting Diagnostic Test for Stage 1 Expert...")
    print(f"📍 Checkpoint: {stage1_ckpt_path}")
    print(f"📅 Target Year: {target_year}")

    # 1. Load the Stage 1 Expert strictly as-is
    # This uses the weights that achieved the high AP on the source year.
    model = SMPModel.load_from_checkpoint(stage1_ckpt_path)
    model.eval()

    # 2. Setup DataModule for the Target Year only
    # We set test_years to the target_year (e.g., 2023) to isolate performance.
    datamodule = FireSpreadDataModule(
        data_dir=data_dir,
        train_years=[], # Not training
        val_years=[],   # Not validating
        test_years=[target_year], 
        batch_size=64,
        num_workers=8
    )

    # 3. Initialize Trainer for Inference
    # We use Trainer.test() to ensure metrics like AP and F1 are logged properly.
    trainer = Trainer(
        accelerator="auto",
        devices=1,
        precision=16, # Match your training precision
        logger=False  # Keep console output clean for diagnostic
    )

    # 4. Execute Test
    print(f"📊 Running evaluation on year {target_year}...")
    results = trainer.test(model, datamodule=datamodule)
    
    print("\n==========================================")
    print(f"🏁 DIAGNOSTIC RESULTS FOR {target_year}")
    print("==========================================")
    for key, value in results[0].items():
        print(f"{key}: {value:.4f}")
    print("==========================================\n")

if __name__ == "__main__":
    # Example for Fold 11 (Target 2023)
    # Replace with your actual paths
    STAGE1_PATH = "/develop/results/wildfire-progression/6l528lvo/checkpoints",
    DATA_DIR = "/develop/data/WildfireSpreadTS_2012_2015_hdf5/"
    TARGET_YEAR = 2023

    diagnostic_test_stage1(STAGE1_PATH, TARGET_YEAR, DATA_DIR)