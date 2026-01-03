import torch
import torch.nn as nn
import torch.nn.functional as F

class CorrelationModule(nn.Module):
    def __init__(self, in_channels=384, out_channels=512):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.query_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.key_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.value_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.out_conv = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        query = self.query_conv(x).view(B, self.out_channels, -1).permute(0, 2, 1)  # [B, H*W, C]
        key = self.key_conv(x).view(B, self.out_channels, -1)  # [B, C, H*W]
        value = self.value_conv(x).view(B, self.out_channels, -1)  # [B, C, H*W]

        attention = torch.bmm(query, key) / (self.out_channels ** 0.5)  # [B, H*W, H*W]
        attention = F.softmax(attention, dim=-1)

        out = torch.bmm(value, attention.permute(0, 2, 1))  # [B, C, H*W]
        out = out.view(B, self.out_channels, H, W)
        out = self.out_conv(out)
        return out

class MultiScaleCorrelation(nn.Module):
    def __init__(self, in_channels_list):
        super().__init__()
        self.correlations = nn.ModuleList([
            CorrelationModule(in_ch, in_ch) for in_ch in in_channels_list
        ])

    def forward(self, features):
        # features: list of feature maps at different scales
        corrs = []
        for i, feat in enumerate(features):
            corr = self.correlations[i](feat)
            # Reduce to 1 channel for correlation bias
            corr = corr.mean(dim=1, keepdim=True)  # (B, 1, H, W)
            corrs.append(corr)
        return corrs
