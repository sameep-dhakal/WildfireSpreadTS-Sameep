import os
import torch
import wandb

from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.utilities import rank_zero_only

from dataloader.FireSpreadDataModule import FireSpreadDataModule
from dataloader.FireSpreadDataset import FireSpreadDataset
from dataloader.utils import get_means_stds_missing_values

from models import SMPModel, BaseModel, ConvLSTMLightning, LogisticRegression  # noqa

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
torch.set_float32_matmul_precision('high')


# =====================================================================
# CUSTOM CLI
# =====================================================================
class MyLightningCLI(LightningCLI):

    # =========================================================
    # STAGE-1 CKPT LOAD DIRS (PRETRAINED UNET)
    # These folders ALREADY contain Stage-1 .ckpt files
    # =========================================================
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


    STAGE1_CKPT_BY_YEAR = {
    2012: "/develop/results/wildfire-progression/gvmlnd1k/checkpoints",
    2013: "/develop/results/wildfire-progression/du8c545g/checkpoints",
    2014: "/develop/results/wildfire-progression/z01uo2hd/checkpoints",
    2015: "/develop/results/wildfire-progression/pqrmt6x3/checkpoints",
    2016: "/develop/results/wildfire-progression/z8qfst4r/checkpoints",
    2017: "/develop/results/wildfire-progression/le2xi881/checkpoints",
    2018: "/develop/results/wildfire-progression/cwdakisi/checkpoints",
    2019: "/develop/results/wildfire-progression/8qhqnjch/checkpoints",
    2020: "/develop/results/wildfire-progression/p6j9l0th/checkpoints",
    2021: "/develop/results/wildfire-progression/2ilct1o4/checkpoints",
    2022: "/develop/results/wildfire-progression/l9a1y4dy/checkpoints",
    2023: "/develop/results/wildfire-progression/pfr5ycr0/checkpoints",
    }

    # =========================================================
    # STAGE-2 SAVE DIRS (WHERE IWAN STAGE-2 CHECKPOINTS WILL BE SAVED)
    # =========================================================
    STAGE2_SAVE_DIR = {
        0: "/develop/results/domain_adaptation_stage2_outputs/fold0",
        1: "/develop/results/domain_adaptation_stage2_outputs/fold1",
        2: "/develop/results/domain_adaptation_stage2_outputs/fold2",
        3: "/develop/results/domain_adaptation_stage2_outputs/fold3",
    }

    def add_arguments_to_parser(self, parser):
        parser.link_arguments("trainer.default_root_dir",
                              "trainer.logger.init_args.save_dir")

        parser.link_arguments("model.class_path",
                              "trainer.logger.init_args.name")

        parser.add_argument("--do_train", type=bool)
        parser.add_argument("--do_predict", type=bool)
        parser.add_argument("--do_test", type=bool)
        parser.add_argument("--do_validate", type=bool, default=False)
        parser.add_argument("--ckpt_path", type=str, default=None)

    # -----------------------------------------------------------------
    # BEFORE MODEL / DATA / TRAINER CREATION
    # -----------------------------------------------------------------
    def before_instantiate_classes(self):
        # determine number of channels
        n_features = FireSpreadDataset.get_n_features(
            self.config.data.n_leading_observations,
            self.config.data.features_to_keep,
            self.config.data.remove_duplicate_features,
        )
        self.config.model.init_args.n_channels = n_features

        # -------------------------------
        # FETCH SOURCE YEAR / FOLD CORRECTLY
        # -------------------------------
        # Align with the datamodule split: use the fold id to get train_years
        data_fold_id = int(self.config.data.data_fold_id)
        train_years, _, _ = FireSpreadDataModule.split_fires(
            data_fold_id,
            self.config.data.additional_data,
        )

        # For domain adaptation we train on all years in the fold (multi-source) and
        # treat the left-out year as the target. Use the fold’s ckpt, not per-year.
        source_year = int(train_years[0])  # representative (used for naming only)
        if self.config.data.additional_data:
            all_years = set(range(2012, 2024))
            missing_years = sorted(list(all_years - set(train_years)))
            if len(missing_years) == 1:
                target_year = missing_years[0]
                self.config.data.target_year = target_year
                # Limit Stage-2 to the inferred target year
                self.config.model.init_args.all_target_years = [target_year]

        # Expose the resolved source_year (representative) in the config
        self.config.data.source_year = source_year
        self.config.model.init_args.source_year = source_year

        _, _, missing_values_rates = get_means_stds_missing_values(train_years)
        fire_rate = 1 - missing_values_rates[-1]
        self.config.model.init_args.pos_class_weight = float(1 / fire_rate)

        # -------------------------------
        # map Stage-1 checkpoint by fold (multi-year source)
        # -------------------------------
        stage1_dir = self.STAGE1_CKPT_DIR.get(data_fold_id)
        if stage1_dir is None:
            raise ValueError(f"No Stage-1 checkpoint dir for fold={data_fold_id}")
        self.config.model.init_args.ckpt_dir = stage1_dir

        print(f"🔥 Using Stage-1 encoder for fold {data_fold_id}:")
        print(f"   {stage1_dir}\n")

        # -------------------------------
        # map Stage-2 save directory
        # -------------------------------
        save_dir = self.STAGE2_SAVE_DIR.get(
            data_fold_id, f"/develop/results/domain_adaptation_stage2_outputs/{data_fold_id}"
        )
        os.makedirs(save_dir, exist_ok=True)

        self.config.trainer.default_root_dir = save_dir
        print(f"💾 Stage-2 checkpoints will be saved to:\n{save_dir}\n")

   
   # -----------------------------------------------------------------
    def before_fit(self):
        self.wandb_setup()

    def before_test(self):
        self.wandb_setup()

    def before_validate(self):
        self.wandb_setup()

    # -----------------------------------------------------------------
    @rank_zero_only
    def wandb_setup(self):
        config_file = os.path.join(wandb.run.dir, "cli_config.yaml")

        cfg = self.parser.dump(self.config, skip_none=False)
        with open(config_file, "w") as f:
            f.write(cfg)

        wandb.save(config_file, policy="now", base_path=wandb.run.dir)

        # wandb.define_metric("train_loss_epoch", summary="min")
        # wandb.define_metric("val_loss", summary="min")
        # wandb.define_metric("train_f1_epoch", summary="max")
        # wandb.define_metric("val_f1", summary="max")
        # wandb.define_metric("val_avg_precision", summary="max")

        wandb.define_metric("train_loss_D_epoch", summary="min")
        wandb.define_metric("train_domain_acc_epoch", summary="max")
        wandb.define_metric("score_D_epoch", summary="min")



# =====================================================================
# MAIN ENTRY
# =====================================================================
def main():

    cli = MyLightningCLI(
        BaseModel,
        FireSpreadDataModule,
        subclass_mode_model=True,
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"parser_mode": "yaml"},
        run=False
    )
    # cli.wandb_setup()

    # if cli.config.do_train:
    #     cli.trainer.fit(cli.model, cli.datamodule,
    #                     ckpt_path=cli.config.ckpt_path)
        
    if cli.config.do_train:
        print("\n⚡ Running IWAN Stage-2 manually (no trainer.fit)…\n")
        cli.model.run_full_iwan(cli.datamodule)
        return  # IMPORTANT: skip all Lightning testing/prediction
    

    ckpt = cli.config.ckpt_path
    if cli.config.do_train:
        ckpt = "best"

    if cli.config.do_validate:
        cli.trainer.validate(cli.model, cli.datamodule, ckpt_path=ckpt)

    if cli.config.do_test:
        cli.trainer.test(cli.model, cli.datamodule, ckpt_path=ckpt)

    if cli.config.do_predict:
        print(f"Loading checkpoint from: {ckpt}")
        preds = cli.trainer.predict(cli.model, cli.datamodule, ckpt_path=ckpt)

        x_af = torch.cat([p[0][:, -1, :, :].squeeze() for p in preds], dim=0)
        y = torch.cat([p[1] for p in preds], dim=0)
        y_hat = torch.cat([p[2] for p in preds], dim=0)

        combined = torch.cat(
            [x_af.unsqueeze(0), y_hat.unsqueeze(0), y.unsqueeze(0)], dim=0
        )

        out_file = os.path.join(
            cli.config.trainer.default_root_dir,
            f"predictions_{wandb.run.id}.pt"
        )
        torch.save(combined, out_file)


if __name__ == "__main__":
    main()



# from pytorch_lightning.utilities import rank_zero_only
# import torch
# from dataloader.FireSpreadDataModule import FireSpreadDataModule
# from pytorch_lightning.cli import LightningCLI
# from models import SMPModel, BaseModel, ConvLSTMLightning, LogisticRegression  # noqa
# from models import BaseModel
# import wandb
# import os

# from dataloader.FireSpreadDataset import FireSpreadDataset
# from dataloader.utils import get_means_stds_missing_values

# os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
# torch.set_float32_matmul_precision('high')


# class MyLightningCLI(LightningCLI):
#     def add_arguments_to_parser(self, parser):
#         parser.link_arguments("trainer.default_root_dir",
#                               "trainer.logger.init_args.save_dir")
#         parser.link_arguments("model.class_path",
#                               "trainer.logger.init_args.name")
#         parser.add_argument("--do_train", type=bool,
#                             help="If True: skip training the model.")
#         parser.add_argument("--do_predict", type=bool,
#                             help="If True: compute predictions.")
#         parser.add_argument("--do_test", type=bool,
#                             help="If True: compute test metrics.")
#         parser.add_argument("--do_validate", type=bool,
#                             default=False, help="If True: compute val metrics.")
#         parser.add_argument("--ckpt_path", type=str, default=None,
#                             help="Path to checkpoint to load for resuming training, for testing and predicting.")

#     def before_instantiate_classes(self):
#         # The number of features is only known inside the data module, but we need that info to instantiate the model.
#         # Since datamodule and model are instantiated at the same time with LightningCLI, we need to set the number of features here.
#         n_features = FireSpreadDataset.get_n_features(
#             self.config.data.n_leading_observations,
#             self.config.data.features_to_keep,
#             self.config.data.remove_duplicate_features)
#         self.config.model.init_args.n_channels = n_features

#         # The exact positive class weight changes with the data fold in the data module, but the weight is needed to instantiate the model.
#         # Non-fire pixels are marked as missing values in the active fire feature, so we simply use that to compute the positive class weight.
#         train_years, _, _ = FireSpreadDataModule.split_fires(
#             self.config.data.data_fold_id, self.config.data.additional_data)
#         _, _, missing_values_rates = get_means_stds_missing_values(train_years)
#         fire_rate = 1 - missing_values_rates[-1]
#         pos_class_weight = float(1 / fire_rate)

#         self.config.model.init_args.pos_class_weight = pos_class_weight

#     def before_fit(self):
#         self.wandb_setup()

#     def before_test(self):
#         self.wandb_setup()

#     def before_validate(self):
#         self.wandb_setup()

#     @rank_zero_only
#     def wandb_setup(self):
#         """
#         Save the config used by LightningCLI to disk, then save that file to wandb.
#         Using wandb.config adds some strange formating that means we'd have to do some 
#         processing to be able to use it again as CLI input.

#         Also define min and max metrics in wandb, because otherwise it just reports the 
#         last known values, which is not what we want.
#         """
#         config_file_name = os.path.join(wandb.run.dir, "cli_config.yaml")

#         cfg_string = self.parser.dump(self.config, skip_none=False)
#         with open(config_file_name, "w") as f:
#             f.write(cfg_string)
#         wandb.save(config_file_name, policy="now", base_path=wandb.run.dir)
#         wandb.define_metric("train_loss_epoch", summary="min")
#         wandb.define_metric("val_loss", summary="min")
#         wandb.define_metric("train_f1_epoch", summary="max")
#         wandb.define_metric("val_f1", summary="max")
#         wandb.define_metric("val_avg_precision", summary="max")


# def main():

#     # LightningCLI automatically creates an argparse parser with required arguments and types,
#     # and instantiates the model and datamodule. For this, it's important to import the model and datamodule classes above.
#     cli = MyLightningCLI(BaseModel, FireSpreadDataModule, subclass_mode_model=True, save_config_kwargs={
#         "overwrite": True}, parser_kwargs={"parser_mode": "yaml"}, run=False)
#     cli.wandb_setup()

#     if cli.config.do_train:
#         cli.trainer.fit(cli.model, cli.datamodule,
#                         ckpt_path=cli.config.ckpt_path)

#     # If we have trained a model, use the best checkpoint for testing and predicting.
#     # Without this, the model's state at the end of the training would be used, which is not necessarily the best.
#     ckpt = cli.config.ckpt_path
#     if cli.config.do_train:
#         ckpt = "best"

#     if cli.config.do_validate:
#         cli.trainer.validate(cli.model, cli.datamodule, ckpt_path=ckpt)

#     if cli.config.do_test:
#         cli.trainer.test(cli.model, cli.datamodule, ckpt_path=ckpt)

#     if cli.config.do_predict:
#         print(f"Loading checkpoint from: {ckpt}")

#         # Produce predictions, save them in a single file, including ground truth fire targets and input fire masks.
#         prediction_output = cli.trainer.predict(
#             cli.model, cli.datamodule, ckpt_path=ckpt)
#         #torch.save(prediction_output, "prediction_output.pt")
#         x_af = torch.cat(
#             list(map(lambda tup: tup[0][:, -1, :, :].squeeze(), prediction_output)), axis=0)
#         y = torch.cat(list(map(lambda tup: tup[1], prediction_output)), axis=0)
#         y_hat = torch.cat(
#             list(map(lambda tup: tup[2], prediction_output)), axis=0)
#         fire_masks_combined = torch.cat(
#             [x_af.unsqueeze(0), y_hat.unsqueeze(0), y.unsqueeze(0)], axis=0)

#         predictions_file_name = os.path.join(
#             cli.config.trainer.default_root_dir, f"predictions_{wandb.run.id}.pt")
#         torch.save(fire_masks_combined, predictions_file_name)


# if __name__ == "__main__":
#     main()
