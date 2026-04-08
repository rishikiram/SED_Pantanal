import timm
import torch
import torch.nn as nn


class CNNBackbone(nn.Module):
    """EfficientNet-B0 encoder that preserves the time dimension.

    Input:  (B, 1, 128, 500)
    Output: (B, C, H', T')  — full spatial feature map, no global pooling
    Output Shape: (B, 1280, 4, 16) in testing with (128, 500) input — note the time dimension is pooled over.
    """

    def __init__(self, model_name: str = 'efficientnet_b0', pretrained: bool = True, in_chans: int = 1):
        super().__init__()
        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,
            global_pool='',  # disable global pooling — keep spatial dims
        )
        # Resolve output channels by doing a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, in_chans, 128, 500)
            out = self.encoder(dummy)
        self.out_channels = out.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, 128, 500)
        Returns: (B, C, H', T')
        """
        return self.encoder(x)
