import torch
import numpy as np
import pandas as pd
import os
import json
from torch.utils.data import DataLoader
from dataset import ForgeryDataset
from utils import rle_encode, postprocess_mask, reciprocal_match_filter, temperature_scale, load_ensemble_models
import cv2

def tta_inference(models, image, device):
    """4-flip TTA inference."""
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

def enhanced_postprocess_mask(prob_mask, features, params):
    """Enhanced postprocessing with reciprocal matching and morphological operations."""
    # Threshold
    binary = (prob_mask > params['threshold']).astype(np.uint8)

    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # Remove small components
    for i in range(1, num_labels):  # Skip background
        if stats[i, cv2.CC_STAT_AREA] < params['a_min']:
            binary[labels == i] = 0

    # Reciprocal match filter (simplified for inference)
    if features is not None:
        binary = reciprocal_match_filter(binary, features, sim_threshold=params['sim_threshold'])

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return binary

def inference_with_ensemble(models, test_loader, device, params):
    """Inference with ensemble and TTA."""
    predictions = {}

    for images, image_ids in test_loader:
        images = images.to(device)
        batch_preds = []

        for i in range(len(images)):
            image = images[i:i+1]  # Single image batch
            pred = tta_inference(models, image, device)
            pred = temperature_scale(pred, params['temperature'])
            batch_preds.append(pred)

        # Stack and average if multiple images
        if len(batch_preds) > 1:
            avg_pred = torch.stack(batch_preds).mean(dim=0)
        else:
            avg_pred = batch_preds[0]

        preds = torch.sigmoid(avg_pred).cpu().numpy()

        for i, pred in enumerate(preds):
            # For simplicity, skip reciprocal matching in inference (features not available)
            pred_mask = enhanced_postprocess_mask(pred[0], None, params)
            rle = rle_encode(pred_mask)
            predictions[image_ids[i]] = rle

    return predictions

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load parameters
    with open('postprocessing_params.json', 'r') as f:
        params = json.load(f)

    # Load top 3 fold models for ensemble
    model_paths = [
        'best_finetuned_fold_1.pth',
        'best_finetuned_fold_2.pth',
        'best_finetuned_fold_3.pth'
    ]
    models = load_ensemble_models(model_paths, device)

    # Load test data
    test_dir = 'recodai-luc-scientific-image-forgery-detection/test_images'
    test_dataset = ForgeryDataset(test_dir, mask_dir=None, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)  # Batch size 1 for simplicity

    # Inference
    predictions = inference_with_ensemble(models, test_loader, device, params)

    # Create submission
    submission = pd.DataFrame.from_dict(predictions, orient='index', columns=['annotation'])
    submission.index.name = 'case_id'
    submission.reset_index(inplace=True)
    submission.to_csv('submission.csv', index=False)
    print("Submission saved to submission.csv")

if __name__ == "__main__":
    main()
