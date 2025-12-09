import os
import torch
import numpy as np
import h5py
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.SMPModel import SMPModel
from dataloader.FireSpreadDataset import FireSpreadDataset
from models.DomainAdpatation.IWANStage2_WeightEstimator import DomainHead3x1024


# -----------------------------------------------------------
# Extract true fire year for each sample in the dataset
# -----------------------------------------------------------
def get_source_years(dataset):
    years = []
    for i in range(len(dataset)):
        yr, _, _ = dataset.find_image_index_from_dataset_index(i)
        years.append(int(yr))
    return np.array(years, dtype=int)


# -----------------------------------------------------------
# Export IWAN Stage-2 weights for a fold
# -----------------------------------------------------------
def export_weights_for_fold(
    fold,
    stage1_ckpt_dir,
    stage2_ckpt_path,
    data_dir,
    save_dir,
    batch_size=32,
    num_workers=4,
    device="cuda"
):
    print(f"\n==============================")
    print(f"🔥 Fold {fold}")
    print(f"Stage-1 ckpt dir:   {stage1_ckpt_dir}")
    print(f"Stage-2 checkpoint: {stage2_ckpt_path}")
    print("==============================\n")

    # Load Stage-1 ckpt file
    ckpts = [f for f in os.listdir(stage1_ckpt_dir) if f.endswith(".ckpt")]
    if len(ckpts) == 0:
        raise RuntimeError(f"No checkpoint found in {stage1_ckpt_dir}")

    stage1_ckpt = os.path.join(stage1_ckpt_dir, ckpts[0])
    print(f"Using Stage-1 ckpt: {stage1_ckpt}")

    # Load Stage-1 SMP Model
    base = SMPModel.load_from_checkpoint(stage1_ckpt)
    encoder = base.model.encoder
    encoder.to(device)
    encoder.eval()

    # Load Stage-2 discriminator
    ckpt2 = torch.load(stage2_ckpt_path, map_location=device)
    feat_dim = ckpt2["feat_dim"]
    target_year = ckpt2["target_year"]

    print(f"Target year for fold {fold}: {target_year}")

    disc = DomainHead3x1024(feat_dim).to(device)
    disc.load_state_dict(ckpt2["discriminator"])
    disc.eval()

    # Build full multi-year source dataset
    all_source_years = list(range(2012, 2024))  # 2012–2023

    source_dataset = FireSpreadDataset(
        data_dir=data_dir,
        included_fire_years=all_source_years,
        n_leading_observations=2,
        crop_side_length=128,
        load_from_hdf5=True,
        is_train=False,
        remove_duplicate_features=True,
        stats_years=all_source_years,
        features_to_keep=None,
        return_doy=False,
    )

    src_loader = DataLoader(
        source_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # True year for every sample
    print("Extracting src_year array...")
    src_years = get_source_years(source_dataset)

    # Compute IWAN weights
    print("Computing importance weights…")
    weights = []
    with torch.no_grad():
        for x, _ in tqdm(src_loader):
            if x.ndim == 5:
                x = x.flatten(1, 2)
            x = x.to(device)

            feats = encoder(x)[-1]
            logits = disc(feats)
            D = torch.sigmoid(logits)      # probability sample is SOURCE
            w = 1.0 - D                    # importance weight

            weights.append(w.cpu().numpy())

    weights = np.concatenate(weights, axis=0)

    # Save HDF5
    out_path = os.path.join(save_dir, f"weights_srcyears_target_{target_year}.h5")
    os.makedirs(save_dir, exist_ok=True)

    print(f"Saving → {out_path}")

    with h5py.File(out_path, "w") as f:
        n = len(source_dataset)
        f.create_dataset("sample_index", data=np.arange(n), compression="gzip")
        f.create_dataset("w", data=weights, compression="gzip")
        f.create_dataset("src_year", data=src_years, compression="gzip")
        f.attrs["target_year"] = int(target_year)

    print(f"✅ Finished fold {fold}: saved {out_path}")


# -----------------------------------------------------------
# MAIN — fold loop
# -----------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Starting Stage-2 weight export job...")

    STAGE1_CKPT_DIR = {
        0: "/develop/results/wildfire-progression/gvuu0kii/checkpoints",
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
        0: "/develop/results/stage2_checkpoints/iwan_stage2_CNNmodel_target_year2012.ckpt",
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
