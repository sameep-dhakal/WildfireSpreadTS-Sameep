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
        # encoder_weights = encoder_weights if encoder_weights != "none" else None
        encoder_weights_for_smp = encoder_weights if encoder_weights in ["imagenet", "ssl", "swsl"] else None


        # self.model = smp.Unet(
        #     encoder_name=encoder_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        #     encoder_weights=encoder_weights,  # use `imagenet` pre-trained weights for encoder initialization
        #     in_channels=n_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        #     classes=1,  # model output channels (number of classes in your dataset)
        # )
        in_channels_total = n_channels
        if use_doy:
            in_channels_total += 1 # adding sin/cos DOY explicitly

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights_for_smp,
            in_channels=in_channels_total,
            classes=1,
)
        
        if encoder_weights == "pastis": 
            primary_ckpt = '/develop/data/utae_pre/model.pth.tar'
            secondary_ckpt = '/src/models/utae_paps_models/model.pth.tar'
            pretrained_checkpoint = primary_ckpt if os.path.exists(primary_ckpt) else secondary_ckpt
            self.load_checkpoint(pretrained_checkpoint)

        elif encoder_weights == "bigearthnet":    
            primary_ckpt = '/develop/data/utae_pre_bigearth/model.pth.tar'
            secondary_ckpt = '/develop/code/WildfireSpreadTS-Sameep/src/models/utae_paps_models/Gallelio-weights/resnet18_bigearthnet_encoder.pth'
            pretrained_checkpoint = primary_ckpt if os.path.exists(primary_ckpt) else secondary_ckpt
            self.load_checkpoint(pretrained_checkpoint)


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

        # previous code how the doys were being calculated 
        # doys = torch.arange(T, device=x.device).unsqueeze(0).repeat(B, 1)
        # print(f" Input shape: {x.shape}, DOYs shape: {doys.shape if doys is not None else 'None'}")

        # code added dy sameep to ajust doy to the data + previous doy + current positions
    # --- Adjust DOYs: use either dummy or anchored modulo 365
        if not self.hparams.use_doy or doys is None:
            relative_positions = torch.arange(T, device=x.device).unsqueeze(0).repeat(B, 1)
        else:
            assert doys.shape == (B, T), f"Expected doys shape {(B,T)}, got {doys.shape}"
            start_doy = doys[:, 0].unsqueeze(1)
            relative_positions = torch.arange(T, device=x.device).unsqueeze(0).repeat(B, 1)
            doys = ((start_doy + relative_positions) % 365 ) / 365.0 # wrap around year

        # # --- Compute sin/cos absolute DOY and add as extra input channels
        # sin_doy = torch.sin(2 * torch.pi * doys / 365).unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1)
        # cos_doy = torch.cos(2 * torch.pi * doys / 365).unsqueeze(-1).unsqueeze(-1)
        # sin_doy = sin_doy.repeat(1, 1, H, W)  # (B, T, H, W)
        # cos_doy = cos_doy.repeat(1, 1, H, W)
        # sin_doy = sin_doy.unsqueeze(2)  # (B, T, 1, H, W)
        # cos_doy = cos_doy.unsqueeze(2)

        # # Concatenate to input channels
        # x = torch.cat([x, sin_doy, cos_doy], dim=2)  # (B, T, C+2, H, W)

        # doy_channel = doys.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, H, W)

        # Concatenate to input
        # x = torch.cat([x, doy_channel], dim=2)  

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
        aggregated_last, attn = self.ltae(last_stage, batch_positions=relative_positions)
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