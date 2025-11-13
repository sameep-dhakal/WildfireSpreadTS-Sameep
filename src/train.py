import os
import torch
import wandb
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.utilities import rank_zero_only

from dataloader.FireSpreadDataModule import FireSpreadDataModule
from dataloader.FireSpreadDataset import FireSpreadDataset
from dataloader.utils import get_means_stds_missing_values
from models import BaseModel, SMPModel, ConvLSTMLightning, LogisticRegression  # noqa

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
torch.set_float32_matmul_precision("high")


# =========================================================
# FOLD → CHECKPOINT AND OUTPUT PATHS
# =========================================================
CHECKPOINT_MAP = {
    0: "/develop/results/wildfire_progression/tc8waap3/checkpoints",
    1: "/develop/results/wildfire_progression/gemkrqiy/checkpoints",
    2: "/develop/results/wildfire_progression/gfosobi8/checkpoints",
    3: "/develop/results/wildfire_progression/cc5fbgta/checkpoints",
}

OUTPUT_ROOT = "/develop/results"


# =========================================================
# CUSTOM CLI CLASS
# =========================================================
class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("trainer.default_root_dir", "trainer.logger.init_args.save_dir")
        parser.link_arguments("model.class_path", "trainer.logger.init_args.name")

        parser.add_argument("--do_train", type=bool, help="If True: train the model.")
        parser.add_argument("--do_predict", type=bool, help="If True: run predictions.")
        parser.add_argument("--do_test", type=bool, help="If True: evaluate on test set.")
        parser.add_argument("--do_validate", type=bool, default=False, help="If True: run validation.")
        parser.add_argument("--ckpt_path", type=str, default=None, help="Checkpoint to load for test/predict.")

    def before_instantiate_classes(self):
        data_cfg = self.config.data
        fold_id = int(data_cfg.data_fold_id)

        # Automatically assign checkpoint directory
        if not hasattr(self.config.model, "init_args"):
            self.config.model.init_args = {}
        ckpt_dir = CHECKPOINT_MAP.get(fold_id)
        if not ckpt_dir or not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(f"❌ Checkpoint for fold {fold_id} not found at {ckpt_dir}")
        self.config.model.init_args.ckpt_dir = ckpt_dir
        print(f"✅ Using checkpoint for fold {fold_id}: {ckpt_dir}")

        # Automatically set output dir
        stage2_dir = os.path.join(OUTPUT_ROOT, f"stage2_fold{fold_id}")
        os.makedirs(stage2_dir, exist_ok=True)
        self.config.trainer.default_root_dir = stage2_dir
        print(f"📁 Output directory: {stage2_dir}")

        # Dynamically set number of input features
        n_features = FireSpreadDataset.get_n_features(
            data_cfg.n_leading_observations,
            data_cfg.features_to_keep,
            data_cfg.remove_duplicate_features,
        )
        self.config.model.init_args.n_channels = n_features

        # Compute class weights based on fire frequency
        train_years, _, _ = FireSpreadDataModule.split_fires(
            data_cfg.data_fold_id, data_cfg.additional_data
        )
        _, _, missing_values_rates = get_means_stds_missing_values(train_years)
        fire_rate = 1 - missing_values_rates[-1]
        pos_class_weight = float(1 / fire_rate)
        self.config.model.init_args.pos_class_weight = pos_class_weight

    def before_fit(self):
        self.wandb_setup()

    def before_test(self):
        self.wandb_setup()

    def before_validate(self):
        self.wandb_setup()

    @rank_zero_only
    def wandb_setup(self):
        """Save CLI config and define summary metrics in wandb."""
        config_file = os.path.join(wandb.run.dir, "cli_config.yaml")
        cfg_str = self.parser.dump(self.config, skip_none=False)
        with open(config_file, "w") as f:
            f.write(cfg_str)
        wandb.save(config_file, policy="now", base_path=wandb.run.dir)

        wandb.define_metric("train_loss_epoch", summary="min")
        wandb.define_metric("val_loss", summary="min")
        wandb.define_metric("train_f1_epoch", summary="max")
        wandb.define_metric("val_f1", summary="max")
        wandb.define_metric("val_avg_precision", summary="max")


# =========================================================
# MAIN FUNCTION
# =========================================================
def main():
    cli = MyLightningCLI(
        BaseModel,
        FireSpreadDataModule,
        subclass_mode_model=True,
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"parser_mode": "yaml"},
        run=False,
    )
    cli.wandb_setup()

    # --- TRAINING ---
    if cli.config.do_train:
        cli.trainer.fit(cli.model, cli.datamodule, ckpt_path=cli.config.ckpt_path)

    # --- CHECKPOINT LOGIC ---
    ckpt = "best" if cli.config.do_train else cli.config.ckpt_path

    # --- VALIDATION / TEST / PREDICT ---
    if cli.config.do_validate:
        cli.trainer.validate(cli.model, cli.datamodule, ckpt_path=ckpt)

    if cli.config.do_test:
        cli.trainer.test(cli.model, cli.datamodule, ckpt_path=ckpt)

    if cli.config.do_predict:
        print(f"Loading checkpoint from: {ckpt}")
        preds = cli.trainer.predict(cli.model, cli.datamodule, ckpt_path=ckpt)
        x_af = torch.cat([t[0][:, -1, :, :].squeeze() for t in preds], axis=0)
        y = torch.cat([t[1] for t in preds], axis=0)
        y_hat = torch.cat([t[2] for t in preds], axis=0)
        fire_masks = torch.cat([x_af.unsqueeze(0), y_hat.unsqueeze(0), y.unsqueeze(0)], axis=0)
        pred_file = os.path.join(cli.config.trainer.default_root_dir, f"predictions_{wandb.run.id}.pt")
        torch.save(fire_masks, pred_file)
        print(f"✅ Predictions saved to: {pred_file}")


if __name__ == "__main__":
    main()
