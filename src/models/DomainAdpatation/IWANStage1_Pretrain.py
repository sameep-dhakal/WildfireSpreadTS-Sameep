# Stage 1 – Source classifier / segmenter (Fs + C)
from typing import Any
import segmentation_models_pytorch as smp
from .BaseModel import BaseModel


class IWANStage1_Pretrain(BaseModel):
    """IWAN Stage 1 – Train source encoder Fs + decoder C on labeled source data."""
    def __init__(
        self,
        encoder_name: str = "resnet18",
        encoder_weights: str = "imagenet",
        n_channels: int = 7,
        flatten_temporal_dimension: bool = True,
        pos_class_weight: float = 1.0,
        loss_function: str = "Focal",
        use_doy: bool = False,
        crop_before_eval: bool = False,
        required_img_size=None,
        alpha_focal=None,
        f1_threshold=None,
        **kwargs: Any,
    ):
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

        self.save_hyperparameters()
        encoder_weights = encoder_weights if encoder_weights != "none" else None
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=n_channels,
            classes=1,
        )
        print(f"✅ Stage 1 – Loaded {encoder_name} encoder ({encoder_weights})")
