import torch
import torch.nn as nn
import torch.nn.functional as F
from .correlation import MultiScaleCorrelation

class ForensicsTransformerHead(nn.Module):
    def __init__(self, in_channels=320, num_heads=8, num_layers=4, hidden_dim=512, out_channels=1):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, hidden_dim, 1)

        # Lightweight transformer: use Conv layers instead of full attention for efficiency
        self.transformer_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim//num_heads),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(num_layers)
        ])

        self.out_conv = nn.Conv2d(hidden_dim, in_channels, 1)

    def forward(self, x, corr_bias=None):
        # x: (B, in_channels, H, W)
        x = self.in_conv(x)  # (B, hidden_dim, H, W)

        for block in self.transformer_blocks:
            residual = x
            x = block(x)
            if corr_bias is not None:
                # Add correlation as bias (upsample if needed)
                corr_bias = F.interpolate(corr_bias, size=x.shape[2:], mode='bilinear', align_corners=False)
                x = x + corr_bias
            x = x + residual  # Residual connection

        out = self.out_conv(x)  # (B, out_channels, H, W)
        return out

class CorrelationGuidedTransformer(nn.Module):
    def __init__(self, in_channels_list, out_channels=1):
        super().__init__()
        self.correlation = MultiScaleCorrelation(in_channels_list)
        self.transformer = ForensicsTransformerHead(in_channels=len(in_channels_list), out_channels=out_channels)

    def forward(self, features):
        corrs = self.correlation(features)
        # Use deepest correlation as bias for transformer
        corr_bias = corrs[-1]  # (B, 1, H, W) at lowest resolution

        # Fuse features for transformer input
        fused = torch.cat([F.interpolate(c, size=features[0].shape[2:], mode='bilinear', align_corners=False) for c in corrs], dim=1)
        # fused: (B, len(in_channels_list), H, W)

        out = self.transformer(fused, corr_bias)
        return out
