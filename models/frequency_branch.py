import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.fft import dctn

class DCTBranch(nn.Module):
    def __init__(self, patch_size=8, num_coeffs=16, hidden_dim=256, out_channels=64):
        super().__init__()
        self.patch_size = patch_size
        self.num_coeffs = num_coeffs
        self.mlp = nn.Sequential(
            nn.Linear(num_coeffs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_channels)
        )

    def forward(self, x):
        # x: (B, 3, H, W)
        B, C, H, W = x.shape
        # Convert to grayscale for DCT
        x_gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]  # (B, H, W)

        # Extract patches
        patches = F.unfold(x_gray.unsqueeze(1), kernel_size=self.patch_size, stride=self.patch_size)  # (B, patch_size^2, num_patches)
        patches = patches.view(B, self.patch_size, self.patch_size, -1).permute(0, 3, 1, 2)  # (B, num_patches, patch_size, patch_size)

        # Compute 2D DCT per patch
        dct_features = []
        for b in range(B):
            batch_dcts = []
            for p in range(patches.shape[1]):
                patch = patches[b, p].cpu().numpy()
                dct_patch = dctn(patch, type=2, norm='ortho')
                # Vectorize low-to-mid coefficients (zigzag or top-left)
                coeffs = dct_patch[:4, :4].flatten()[:self.num_coeffs]  # First 16 coeffs
                batch_dcts.append(coeffs)
            dct_features.append(torch.tensor(np.array(batch_dcts), dtype=torch.float32, device=x.device))

        dct_features = torch.stack(dct_features)  # (B, num_patches, num_coeffs)

        # Pass through MLP
        embeddings = self.mlp(dct_features)  # (B, num_patches, out_channels)

        # Reshape to spatial map (assume square grid)
        grid_size = int(np.sqrt(embeddings.shape[1]))
        spatial_map = embeddings.view(B, grid_size, grid_size, out_channels).permute(0, 3, 1, 2)  # (B, out_channels, grid_size, grid_size)

        # Upsample to match backbone scales (e.g., 1/8 resolution)
        spatial_map = F.interpolate(spatial_map, size=(H//8, W//8), mode='bilinear', align_corners=False)
        return spatial_map

class PhaseCorrelationBranch(nn.Module):
    def __init__(self, out_channels=64):
        super().__init__()
        self.conv = nn.Conv2d(1, out_channels, 3, padding=1)

    def forward(self, x):
        # x: (B, 3, H, W)
        # Simplified: compute FFT phase correlation on sliding windows
        # For efficiency, compute on downsampled version
        x_down = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x_gray = 0.299 * x_down[:, 0] + 0.587 * x_down[:, 1] + 0.114 * x_down[:, 2]  # (B, H/2, W/2)

        # Compute FFT
        fft = torch.fft.fft2(x_gray)
        phase = torch.angle(fft)

        # Compute phase correlation (simplified: magnitude of phase difference)
        # In practice, correlate with reference or neighboring windows
        # For now, use phase magnitude as feature
        phase_feat = torch.abs(phase).unsqueeze(1)  # (B, 1, H/2, W/2)

        # Convolve to get features
        features = self.conv(phase_feat)
        features = self.norm(features)
        features = F.interpolate(features, size=(x.shape[2]//8, x.shape[3]//8), mode='bilinear', align_corners=False)
        return features
