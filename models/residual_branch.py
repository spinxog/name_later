import torch
import torch.nn as nn
import torch.nn.functional as F

class SRMConv2d(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=5, stride=1, padding=2):
        super().__init__()
        # SRM kernels (Bayar & Stamm, sum to zero)
        self.kernels = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size))
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize SRM-like high-pass filters
        with torch.no_grad():
            # Kernel 1: horizontal edges
            self.kernels[0, 0, 2, 1:4] = torch.tensor([-1, 2, -1])
            # Kernel 2: vertical edges
            self.kernels[1, 0, 1:4, 2] = torch.tensor([-1, 2, -1])
            # Kernel 3: diagonal
            self.kernels[2, 0, 1, 1] = -1
            self.kernels[2, 0, 2, 2] = 2
            self.kernels[2, 0, 3, 3] = -1
            # Additional kernels if needed
            if self.kernels.shape[0] > 3:
                # Kernel 4: cross pattern
                self.kernels[3, 0, 1:4, 1:4] = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
                # Kernel 5: corner
                self.kernels[4, 0, 0, 0] = -1
                self.kernels[4, 0, 0, 4] = -1
                self.kernels[4, 0, 4, 0] = -1
                self.kernels[4, 0, 4, 4] = -1
                self.kernels[4, 0, 2, 2] = 4

    def forward(self, x):
        return F.conv2d(x, self.kernels, padding=self.kernels.shape[-1]//2)

class ResidualEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((32, 32))  # Downsample to 1/8 scale assuming 256x256 input
        )

    def forward(self, x):
        return self.encoder(x)

class NoiseResidualBranch(nn.Module):
    def __init__(self, num_kernels=5, kernel_size=5):
        super().__init__()
        self.srm_conv = SRMConv2d(out_channels=num_kernels, kernel_size=kernel_size)
        self.residual_encoder = ResidualEncoder(in_channels=num_kernels)

    def forward(self, x):
        # x: (B, 3, H, W)
        residuals = self.srm_conv(x)  # (B, num_kernels, H, W)
        features = self.residual_encoder(residuals)  # (B, 64, 32, 32)
        return features
