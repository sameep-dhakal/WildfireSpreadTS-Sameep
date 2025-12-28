"""
Launcher for IWANJointSegmentation with FireSpreadDataModule.
- Loads Stage-1 checkpoint per fold to build frozen Fs and trainable Ft.
- Requires FireSpreadDataModule to provide target_dataloader() for target batches.
"""
import os
import torch
import wandb

from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.utilities import rank_zero_only

from dataloader.FireSpreadDataModule import FireSpreadDataModule
from dataloader.FireSpreadDataset import FireSpreadDataset
from dataloader.utils import get_means_stds_missing_values

from models import BaseModel  # LightningCLI will load subclasses via class_path
from models.DomainAdpatation.IWANJointSegmentation import IWANJointSegmentation

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
torch.set_float32_matmul_precision("high")

DEFAULT_STAGE_FEATURES = [0, 1, 2, 3, 4, 38, 39]


def first_ckpt(path):
    """Return the first .ckpt in a directory."""
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Checkpoint directory not found: {path}")
    files = [f for f in os.listdir(path) if f.endswith(".ckpt")]
    if not files:
        raise FileNotFoundError(f"No .ckpt files in {path}")
    return os.path.join(path, sorted(files)[0])


class MyLightningCLI(LightningCLI):
    # Stage-1 checkpoints by fold
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

    def add_arguments_to_parser(self, parser):
        parser.link_arguments("trainer.default_root_dir", "trainer.logger.init_args.save_dir")
        parser.link_arguments("model.class_path", "trainer.logger.init_args.name")
        parser.add_argument("--do_train", type=bool)
        parser.add_argument("--do_predict", type=bool)
        parser.add_argument("--do_test", type=bool)
        parser.add_argument("--do_validate", type=bool, default=False)
        parser.add_argument("--ckpt_path", type=str, default=None)

    def before_instantiate_classes(self):
        # Use default feature subset for IWAN if none provided
        if self.config.data.features_to_keep is None:
            self.config.data.features_to_keep = DEFAULT_STAGE_FEATURES

        n_features = FireSpreadDataset.get_n_features(
            self.config.data.n_leading_observations,
            self.config.data.features_to_keep,
            self.config.data.remove_duplicate_features,
        )
        self.config.model.init_args.n_channels = n_features

        data_fold_id = int(self.config.data.data_fold_id)
        train_years, _, _ = FireSpreadDataModule.split_fires(
            data_fold_id,
            self.config.data.additional_data,
        )
        source_year = int(train_years[0])
        self.config.data.source_year = source_year
        self.config.model.init_args.stage1_ckpt = first_ckpt(self.STAGE1_CKPT_DIR[data_fold_id])

        # optional target year hint (not used directly by IWANJoint but good for bookkeeping)
        if self.config.data.additional_data:
            all_years = set(range(2012, 2024))
            missing_years = sorted(list(all_years - set(train_years)))
            target_year = missing_years[0] if len(missing_years) == 1 else None
            self.config.data.target_year = target_year

        _, _, missing_values_rates = get_means_stds_missing_values(train_years)
        fire_rate = 1 - missing_values_rates[-1]
        self.config.model.init_args.pos_class_weight = float(1 / fire_rate)

        # ensure output dir exists
        save_dir = self.config.trainer.default_root_dir or "/develop/results/"
        os.makedirs(save_dir, exist_ok=True)
        print(f"🔥 Stage-1 ckpt: {self.config.model.init_args.stage1_ckpt}")
        print(f"💾 Outputs: {save_dir}\n")

    def before_fit(self):
        self._wandb_setup()

    def before_test(self):
        self._wandb_setup()

    def before_validate(self):
        self._wandb_setup()

    @rank_zero_only
    def _wandb_setup(self):
        if wandb.run is None:
            return
        config_file = os.path.join(wandb.run.dir, "cli_config.yaml")
        cfg = self.parser.dump(self.config, skip_none=False)
        with open(config_file, "w") as f:
            f.write(cfg)
        wandb.save(config_file, policy="now", base_path=wandb.run.dir)


def main():
    cli = MyLightningCLI(
        BaseModel,
        FireSpreadDataModule,
        subclass_mode_model=True,
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"parser_mode": "yaml"},
        run=False,
    )

    ckpt = cli.config.ckpt_path or None
    if cli.config.do_train:
        cli.trainer.fit(cli.model, cli.datamodule, ckpt_path=cli.config.ckpt_path)
        ckpt = None  # continue with in-memory weights

    if cli.config.do_validate:
        cli.trainer.validate(cli.model, cli.datamodule, ckpt_path=ckpt)

    if cli.config.do_test:
        cli.trainer.test(cli.model, cli.datamodule, ckpt_path=ckpt)

    if cli.config.do_predict:
        preds = cli.trainer.predict(cli.model, cli.datamodule, ckpt_path=ckpt)
        x_af = torch.cat([p[0][:, -1, :, :].squeeze() for p in preds], dim=0)
        y = torch.cat([p[1] for p in preds], dim=0)
        y_hat = torch.cat([p[2] for p in preds], dim=0)
        combined = torch.cat([x_af.unsqueeze(0), y_hat.unsqueeze(0), y.unsqueeze(0)], dim=0)
        out_file = os.path.join(cli.config.trainer.default_root_dir, f"predictions_{wandb.run.id}.pt")
        torch.save(combined, out_file)


if __name__ == "__main__":
    main()
