from typing import Any

import torch
import segmentation_models_pytorch as smp
from models.utae_paps_models.ltae import LTAE2d
from models.utae_paps_models.utae import Temporal_Aggregator
import os

from models.BaseModel import BaseModel


class SMPTempModel(BaseModel):
    """_summary_ Segmentation model based on the SMP package. We add an LTAE block to the U-Net model.
    """
    def __init__(
        self,
        encoder_name: str,
        n_channels: int,
        flatten_temporal_dimension: bool,
        pos_class_weight: float,
        encoder_weights = None,
        use_doy: bool = False,
        *args: Any,
        **kwargs: Any
    ):
        super().__init__(
            n_channels=n_channels,
            flatten_temporal_dimension=flatten_temporal_dimension,
            pos_class_weight=pos_class_weight,
            use_doy=use_doy, 
            *args,
            **kwargs
        )
        self.save_hyperparameters()
        encoder_weights = encoder_weights if encoder_weights != "none" else None

        if encoder_weights == "pastis": 
            primary_ckpt = '/develop/data/utae_pre/model.pth.tar'
            secondary_ckpt = '/home/sl221120/WildfireSpreadTS/src/models/utae_paps_models/model.pth.tar'
            pretrained_checkpoint = primary_ckpt if os.path.exists(primary_ckpt) else secondary_ckpt
            self.load_checkpoint(pretrained_checkpoint)

        self.model = smp.Unet(
            encoder_name=encoder_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=encoder_weights,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=n_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,  # model output channels (number of classes in your dataset)
        )
        self.last_stage_channels = self.model.encoder.out_channels[-1]
        self.ltae = LTAE2d(
            in_channels=self.last_stage_channels,
            n_head=16,
            d_k=4,
            mlp=[256, self.last_stage_channels],
            dropout=0.2,
            d_model=256,
            T=1000,
            return_att=True,
            positional_encoding=True,
        )
        self.temporal_aggregator = Temporal_Aggregator(mode='att_group')
        
        print(f"Loaded {encoder_name} with {encoder_weights} weights + LTAE")


    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load a pretrained checkpoint for the model.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint["state_dict"]
        prefix = "encoder."
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith(prefix):
                new_state_dict[key[len(prefix):]] = value
            else:
                new_state_dict[key] = value
        model_state_dict = self.model.state_dict()
        filtered_state_dict = {
            k: v
            for k, v in new_state_dict.items()
            if k in model_state_dict and model_state_dict[k].size() == v.size()
        }
        # Load the weights into the model
        self.model.load_state_dict(filtered_state_dict, strict=False)
        print(f"Checkpoint loaded successfully from '{checkpoint_path}'")


    def forward(self, x: torch.Tensor, doys: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
            # ✅ Properly use DOY passed from dataset if available and if self.use_doy is True
        if not self.hparams.use_doy or doys is None:
            doys = torch.arange(T, device=x.device).unsqueeze(0).repeat(B, 1)
            print(f"🚀 Using dummy positional encoding: {doys[0]}")

        else:
            assert doys.shape == (B, T), f"Expected doys shape {(B,T)}, got {doys.shape}"
            print(f"🚀 Using real DOY from dataset: {doys[0]}")


        # ✅ Optional debug print to see what is actually going to LTAE
        print(f"🚀 DOYs being passed to LTAE: {doys[0]}")

        num_stages = len(self.model.encoder.out_channels)
        encoder_features = [[] for _ in range(num_stages)]
        # Extract encoder features for each time step
        for t in range(T):
            x_t = x[:, t, :, :, :]
            features = self.model.encoder(x_t)
            for i in range(num_stages):
                encoder_features[i].append(features[i])
        # Process the last stage with LTAE
        last_stage = torch.stack(encoder_features[-1], dim=1)  # (B, T, C, H, W)
        aggregated_last, attn = self.ltae(last_stage, batch_positions=doys)
        # Process other stages with Temporal Aggregator
        aggregated_skips = []
        n_heads = 16
        for i in range(1, num_stages - 1):
            stage = torch.stack(encoder_features[i], dim=1)  # (B, T, C_i, H_i, W_i)
            B_i, T_i, C_i, H_i, W_i = stage.shape
            # Now aggregate over time using the attention mask from LTAE
            aggregated = self.temporal_aggregator(stage, attn_mask=attn)
            aggregated_skips.append(aggregated)
        dummy = encoder_features[0][0]
        decoder_features = [dummy] + aggregated_skips + [aggregated_last]
        decoder_output = self.model.decoder(*decoder_features)
        masks = self.model.segmentation_head(decoder_output)
        return masks