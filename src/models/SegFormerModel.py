import torch
import torch.nn as nn
from transformers import SegformerModel, SegformerConfig
from models.BaseModel import BaseModel

import os
os.environ["DISABLE_FLASH_ATTENTION"] = "1"



class SegFormerModel(BaseModel):
    """
    SegFormer segmentation model (MiT backbone, pretrained on ImageNet/ADE20K)
    Drop-in replacement for SMPModel / SMPTempModel.
    """

    def __init__(
        self,
        model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        n_channels: int = 3,
        flatten_temporal_dimension: bool = False,
        pos_class_weight: float = 1.0,
        *args,
        **kwargs
    ):
        super().__init__(
            n_channels=n_channels,
            flatten_temporal_dimension=flatten_temporal_dimension,
            pos_class_weight=pos_class_weight,
            *args,
            **kwargs,
        )

        self.save_hyperparameters()

        # ------------- Load pretrained SegFormer backbone -------------
        config = SegformerConfig.from_pretrained(model_name)
        self.backbone = SegformerModel.from_pretrained(model_name, config=config)

        # ------------- Modify first patch embedding for n_channels -------------
        if n_channels != 3:
            old_proj = self.backbone.encoder.patch_embeddings[0].proj
            new_proj = nn.Conv2d(
                n_channels,
                old_proj.out_channels,
                kernel_size=old_proj.kernel_size,
                stride=old_proj.stride,
                padding=old_proj.padding,
                bias=old_proj.bias,
            )
            nn.init.kaiming_normal_(new_proj.weight, mode="fan_out", nonlinearity="relu")
            self.backbone.encoder.patch_embeddings[0].proj = new_proj
            print(f"SegFormer input projection modified: {n_channels} → {old_proj.out_channels}")

        # ------------- Segmentation head -------------
        decoder_in = config.hidden_sizes[-1]
        self.decode_head = nn.Sequential(
            nn.Conv2d(decoder_in, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1)  # binary segmentation
        )

        print(f"Loaded SegFormer backbone: {model_name} (ImageNet-pretrained)")

    # ---------------------------------------------------------------
    # forward pass
    # ---------------------------------------------------------------
    def forward(self, x, doys=None):
        # Flatten temporal dimension if needed (same as SMPTempModel)
        if self.hparams.flatten_temporal_dimension and x.ndim == 5:
            B, T, C, H, W = x.shape
            x = x.view(B, T * C, H, W)

        outputs = self.backbone(x)
        features = outputs.last_hidden_state  # [B, C, H/32, W/32]

        logits = self.decode_head(features)
        logits = nn.functional.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        return logits
