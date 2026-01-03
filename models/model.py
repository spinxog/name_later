import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import Backbone
from .correlation import CorrelationModule
from .forensics_transformer import ForensicsTransformerHead
from .decoder import UNetDecoder
from .residual_branch import NoiseResidualBranch
from .frequency_branch import DCTBranch, PhaseCorrelationBranch

class ForgeryDetectionModel(nn.Module):
    def __init__(self, backbone_name='efficientnet_b0', head_type='correlation_transformer'):
        super().__init__()
        self.backbone = Backbone(backbone_name)
        self.correlation = CorrelationModule(in_channels=self.backbone.out_channels[-1]) if 'correlation' in head_type else None
        self.transformer = ForensicsTransformerHead(in_channels=self.backbone.out_channels[-1]) if 'transformer' in head_type else None
        self.decoder = UNetDecoder(self.backbone.out_channels)

    def forward(self, x):
        features = self.backbone(x)
        if self.correlation:
            corr_features = self.correlation(features[-1])  # Use deepest feature for correlation
            if self.transformer:
                # Integrate correlation into transformer
                features[-1] = self.transformer(features[-1], corr_features)
            else:
                # Simple concatenation or addition
                features[-1] = features[-1] + corr_features.unsqueeze(1).expand_as(features[-1])
        elif self.transformer:
            features[-1] = self.transformer(features[-1])

        out = self.decoder(features)
        # Upsample to 512x512 if necessary
        if out.shape[2:] != (512, 512):
            out = F.interpolate(out, size=(512, 512), mode='bilinear', align_corners=False)
        return out
