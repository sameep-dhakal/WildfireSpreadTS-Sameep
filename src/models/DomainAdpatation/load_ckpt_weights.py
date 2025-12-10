import os
import torch
import numpy as np
import h5py
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.SMPModel import SMPModel
from dataloader.FireSpreadDataset import FireSpreadDataset
from models.DomainAdpatation.IWANStage2_WeightEstimator import DomainHead3x1024



# =====================================================================
# Inspect checkpoints (optional helper)
# =====================================================================
def inspect_checkpoint(ckpt_path):
    print("\n========================================")
    print(f"🔍 Inspecting checkpoint:\n{ckpt_path}")
    print("========================================")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    if "discriminator" not in ckpt:
        print("❌ No 'discriminator' key in checkpoint!")
        print("Available keys:", ckpt.keys())
        return

    disc_state = ckpt["discriminator"]

    print("\n📌 Discriminator State Dict Keys:")
    for k, v in disc_state.items():
        print(f"  {k:40s} {tuple(v.shape)}")

    print("\n🎯 Feature Dim:", ckpt.get("feat_dim", "NOT FOUND"))
    print("🎯 Target Year:", ckpt.get("target_year", "NOT FOUND"))

    print("========================================\n")



# =====================================================================
# Extract true fire year for each sample in the dataset
# =====================================================================
def get_source_years(dataset):
    years = []
    for i in range(len(dataset)):
        yr, _, _ = dataset.find_image_index_from_dataset_index(i)
        years.append(int(yr))
    return np.array(years, dtype=int)



# =====================================================================
# EXPORT FUNCTION — FINAL, CORRECTED, 7-CHANNEL CONSISTENT
# =====================================================================
def export_weights_for_fold(
    fold,
    stage1_ckpt_dir,
    stage2_ckpt_path,
    data_dir,
    save_dir,
    batch_size=64,
    num_workers=4,
    device="cuda"
):
    print("\n==============================")
    print(f"🔥 Exporting Stage-2 Weights: Fold {fold}")
    print(f"Stage-1 ckpt dir:   {stage1_ckpt_dir}")
    print(f"Stage-2 checkpoint: {stage2_ckpt_path}")
    print("==============================\n")

    # ---------------------------------------------------
    # Load Stage-1 segmentation model
    # ---------------------------------------------------
    ckpts = [f for f in os.listdir(stage1_ckpt_dir) if f.endswith(".ckpt")]
    if len(ckpts) == 0:
        raise RuntimeError(f"No checkpoint found in {stage1_ckpt_dir}")

    stage1_ckpt = os.path.join(stage1_ckpt_dir, ckpts[0])
    print(f"Using Stage-1 ckpt: {stage1_ckpt}")

    base = SMPModel.load_from_checkpoint(stage1_ckpt)
    encoder = base.model.encoder
    encoder.to(device)
    encoder.eval()

    # Stage-1 true input channels (should be 7)
    true_stage1_in_channels = base.hparams.n_channels
    print(f"Stage-1 expected input channels = {true_stage1_in_channels}")

    # ---------------------------------------------------
    # Load Stage-2 discriminator
    # ---------------------------------------------------
    ckpt2 = torch.load(stage2_ckpt_path, map_location=device)
    feat_dim = ckpt2["feat_dim"]
    target_year = ckpt2["target_year"]

    print(f"Target year for fold {fold}: {target_year}")

    disc = DomainHead3x1024(feat_dim).to(device)
    disc.load_state_dict(ckpt2["discriminator"])
    disc.eval()

    # ---------------------------------------------------
    # BUILD DATASET EXACTLY LIKE STAGE-1
    # MUST output 7 channels
    # ---------------------------------------------------
    features_to_keep = [0, 1, 2, 3, 4, 38, 39]  # EXACT STAGE-1 FEATURES

    source_dataset = FireSpreadDataset(
        data_dir=data_dir,
        included_fire_years=list(range(2012, 2024)),
        n_leading_observations=1,   # Stage-1 was monotemporal
        crop_side_length=128,
        load_from_hdf5=True,
        is_train=False,

        # CRITICAL: identical preprocessing to Stage-1
        remove_duplicate_features=True,
        features_to_keep=features_to_keep,
        additional_data=True,
        return_doy=False,
    )

    # Validate dataset output matches Stage-1 input channels
    test_x, _ = source_dataset[0]
    C = test_x.shape[0] if test_x.ndim == 3 else test_x.shape[1]

    print("Dataset sample shape AFTER preprocessing:", test_x.shape)

    if C != true_stage1_in_channels:
        raise RuntimeError(
            f"Dataset produces {C} channels but Stage-1 encoder expects {true_stage1_in_channels}. "
            f"Check remove_duplicate_features/features_to_keep settings."
        )

    src_loader = DataLoader(
        source_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # ---------------------------------------------------
    # Collect true source years
    # ---------------------------------------------------
    print("Extracting src_year array…")
    src_years = get_source_years(source_dataset)

    # ---------------------------------------------------
    # Weight computation
    # ---------------------------------------------------
    print("Computing IWAN importance weights…")
    weights = []

    with torch.no_grad():
        for x, _ in tqdm(src_loader):
            x = x.to(device)

            feats = encoder(x)[-1]      # Deepest feature map
            logits = disc(feats)
            D = torch.sigmoid(logits)   # P(sample = source)
            w = 1.0 - D                 # IWAN importance

            weights.append(w.cpu().numpy())

    weights = np.concatenate(weights, axis=0)

    # ---------------------------------------------------
    # Save output HDF5
    # ---------------------------------------------------
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"weights_srcyears_target_{target_year}.h5")

    print(f"Saving → {out_path}")

    with h5py.File(out_path, "w") as f:
        f.create_dataset("sample_index", data=np.arange(len(source_dataset)))
        f.create_dataset("w", data=weights)
        f.create_dataset("src_year", data=src_years)
        f.attrs["target_year"] = int(target_year)

    print(f"✅ Fold {fold} exported successfully.\n")



# =====================================================================
# MAIN — Loop over folds
# =====================================================================
if __name__ == "__main__":
    print("🚀 Starting Stage-2 weight export job...")

    STAGE1_CKPT_DIR = {
        1: "/develop/results/wildfire-progression/hvsbcl8a/checkpoints",
        2: "/develop/results/wildfire-progression/zsqvrj9f/checkpoints",
        3: "/develop/results/wildfire-progression/04zyztgf/checkpoints",
        4: "/develop/results/wildfire-progression/kf5e2z7i/checkpoints",
        5: "/develop/results/wildfire-progression/z4fpm67c/checkpoints",
        6: "/develop/results/wildfire-progression/xu8nf4pr/checkpoints",
        7: "/develop/results/wildfire-progression/3gx6vn1b/checkpoints",
        8: "/develop/results/wildfire-progression/gbco149i/checkpoints",
        9: "/develop/results/wildfire-progression/u2jopt8y/checkpoints",
        10: "/develop/results/wildfire-progression/asvp9e1m/checkpoints",
        11: "/develop/results/wildfire-progression/6l528lvo/checkpoints",
        12: "/develop/results/wildfire-progression/o7t67rsw/checkpoints",
    }

    STAGE2_CKPT_DIR = {
        1: "/develop/results/stage2_checkpoints/iwan_stage2_CNNmodel_target_year2013.ckpt",
        2: "/develop/results/stage2_checkpoints/iwan_stage2_CNNmodel_target_year2014.ckpt",
        3: "/develop/results/stage2_checkpoints/iwan_stage2_CNNmodel_target_year2015.ckpt",
        4: "/develop/results/stage2_checkpoints/iwan_stage2_CNNmodel_target_year2016.ckpt",
        5: "/develop/results/stage2_checkpoints/iwan_stage2_CNNmodel_target_year2017.ckpt",
        6: "/develop/results/stage2_checkpoints/iwan_stage2_CNNmodel_target_year2018.ckpt",
        7: "/develop/results/stage2_checkpoints/iwan_stage2_CNNmodel_target_year2019.ckpt",
        8: "/develop/results/stage2_checkpoints/iwan_stage2_CNNmodel_target_year2020.ckpt",
        9: "/develop/results/stage2_checkpoints/iwan_stage2_CNNmodel_target_year2021.ckpt",
        10: "/develop/results/stage2_checkpoints/iwan_stage2_CNNmodel_target_year2022.ckpt",
        11: "/develop/results/stage2_checkpoints/iwan_stage2_CNNmodel_target_year2023.ckpt",
    }

    DATA_DIR = "/develop/data/WildfireSpreadTS_2012_2015_hdf5/"
    SAVE_DIR = "/develop/results/stage2_weight_exports/"

    for fold in STAGE1_CKPT_DIR:
        export_weights_for_fold(
            fold=fold,
            stage1_ckpt_dir=STAGE1_CKPT_DIR[fold],
            stage2_ckpt_path=STAGE2_CKPT_DIR[fold],
            data_dir=DATA_DIR,
            save_dir=SAVE_DIR,
        )

    print("\n🎉 ALL FOLDS COMPLETED SUCCESSFULLY!")
