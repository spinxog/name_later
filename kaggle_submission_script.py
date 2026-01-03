#!/usr/bin/env python3
"""
RECOD AI LUC Scientific Image Forgery Detection - Submission Script
This script implements the full inference pipeline for Kaggle submission.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
import timm

# Disable internet access
os.environ['WANDB_MODE'] = 'offline'

# Set deterministic behavior
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Model Architecture (copied and adapted for submission)
class SwinBackbone(nn.Module):
    def __init__(self, model_name='swin_tiny_patch4_window7_224', out_indices=(1, 2, 3)):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, features_only=True, out_indices=out_indices)

    def forward(self, x):
        return self.backbone(x)

class CorrelationModule(nn.Module):
    def __init__(self, embed_dim=384, num_heads=6):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, features):
        B, C, H, W = features.shape
        N = H * W
        features_flat = features.view(B, C, N).permute(0, 2, 1)  # [B, N, C]

        qkv = self.qkv_proj(features_flat)
        q, k, v = qkv.chunk(3, dim=-1)

        # Self-attention with correlation
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, v)
        output = self.out_proj(attn_output)

        return output.view(B, H, W, C).permute(0, 3, 1, 2)

class Decoder(nn.Module):
    def __init__(self, in_channels=384, out_channels=1):
        super().__init__()
        self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)

class ForgeryDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SwinBackbone()
        self.correlation = CorrelationModule()
        self.decoder = Decoder()

    def forward(self, x):
        features = self.backbone(x)[2]  # Use last feature map
        features = self.correlation(features)
        output = self.decoder(features)
        return output

# Load models
def load_ensemble_models(model_paths, device):
    models = []
    for path in model_paths:
        model = ForgeryDetectionModel()
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)
    return models

# Dataset for test images
class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']

        return image, self.image_files[idx].replace('.png', '')

# Transforms (same as validation)
test_transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# RLE encoding (exact copy from competition notebook)
def rle_encode(mask):
    """Run-length encoding for binary mask."""
    pixels = mask.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    if len(runs) == 0:
        return "authentic"
    return " ".join(str(x) for x in runs)

# Temperature scaling
def temperature_scale(logits, temperature):
    return logits / temperature

# TTA inference
def tta_inference(models, image, device):
    tta_transforms = [
        lambda x: x,  # Original
        lambda x: torch.flip(x, [2]),  # Horizontal flip
        lambda x: torch.flip(x, [3]),  # Vertical flip
        lambda x: torch.flip(torch.flip(x, [2]), [3])  # Both flips
    ]
    inv_transforms = [
        lambda x: x,
        lambda x: torch.flip(x, [2]),
        lambda x: torch.flip(x, [3]),
        lambda x: torch.flip(torch.flip(x, [2]), [3])
    ]

    preds = []
    for model in models:
        model_preds = []
        for transform, inv_transform in zip(tta_transforms, inv_transforms):
            aug_image = transform(image)
            with torch.no_grad():
                pred = model(aug_image)
                pred = inv_transform(pred)  # Inverse transform
            model_preds.append(pred)
        # Average TTA for this model
        preds.append(torch.stack(model_preds).mean(dim=0))

    # Average across models
    avg_pred = torch.stack(preds).mean(dim=0)
    return avg_pred

# Postprocessing
def postprocess_mask(prob_mask, threshold=0.45, min_area=8):
    mask = (prob_mask > threshold).astype(np.uint8)
    if mask.sum() == 0:
        return mask

    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            mask[labels == i] = 0

    return mask

# Main inference
if __name__ == "__main__":
    # Load parameters
    with open('/kaggle/input/postprocessing-params/postprocessing_params.json', 'r') as f:
        params = json.load(f)

    # Load models
    model_paths = [
        '/kaggle/input/model-weights/best_finetuned_fold_0.pth',
        '/kaggle/input/model-weights/best_finetuned_fold_1.pth',
        '/kaggle/input/model-weights/best_finetuned_fold_2.pth',
        '/kaggle/input/model-weights/best_finetuned_fold_3.pth',
        '/kaggle/input/model-weights/best_finetuned_fold_4.pth'
    ]
    models = load_ensemble_models(model_paths, device)

    # Dataset
    test_dataset = TestDataset('/kaggle/input/recodai-luc-scientific-image-forgery-detection/test_images', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    # Inference
    predictions = {}
    for images, image_ids in tqdm(test_loader, desc="Inference"):
        images = images.to(device)

        # Full pipeline
        pred = tta_inference(models, images, device)
        pred = temperature_scale(pred, params['temperature'])
        pred = torch.sigmoid(pred).cpu().numpy()[0, 0]

        # Postprocess
        processed = postprocess_mask(pred, threshold=params['threshold'], min_area=params['a_min'])

        # Encode
        if processed.sum() == 0:
            predictions[image_ids[0]] = 'authentic'
        else:
            predictions[image_ids[0]] = rle_encode(processed)

    # Create submission
    submission = pd.DataFrame({
        'case_id': list(predictions.keys()),
        'annotation': list(predictions.values())
    })

    # Sort by case_id
    submission = submission.sort_values('case_id')
    submission.to_csv('submission.csv', index=False)

    # Summary
    authentic_count = sum(1 for v in predictions.values() if v == 'authentic')
    forged_count = len(predictions) - authentic_count
    print(f"Predictions: {authentic_count} authentic, {forged_count} forged")
    print("Submission saved to submission.csv")

    # Display first 5 rows
    print("\nFirst 5 rows of submission:")
    print(submission.head())
