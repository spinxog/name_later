import torch
import time
import numpy as np
from torch.utils.data import DataLoader
from dataset import ForgeryDataset
from utils import load_ensemble_models, temperature_scale
import json

def tta_inference_runtime(models, image, device):
    """4-flip TTA inference for runtime measurement."""
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

def measure_inference_time(num_images=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load parameters
    with open('postprocessing_params.json', 'r') as f:
        params = json.load(f)

    # Load top 3 fold models
    model_paths = [
        'best_finetuned_fold_1.pth',
        'best_finetuned_fold_2.pth',
        'best_finetuned_fold_3.pth'
    ]
    models = load_ensemble_models(model_paths, device)

    # Load test data (subset for measurement)
    test_dir = 'recodai-luc-scientific-image-forgery-detection/test_images'
    test_dataset = ForgeryDataset(test_dir, mask_dir=None, is_train=False)

    # Use first num_images
    subset_dataset = torch.utils.data.Subset(test_dataset, range(min(num_images, len(test_dataset))))
    test_loader = DataLoader(subset_dataset, batch_size=1, shuffle=False, num_workers=1)

    times = []
    for images, image_ids in test_loader:
        images = images.to(device)
        start_time = time.time()

        # Full inference pipeline
        pred = tta_inference_runtime(models, images, device)
        pred = temperature_scale(pred, params['temperature'])

        end_time = time.time()
        times.append(end_time - start_time)

    avg_time = np.mean(times)
    total_test_images = 45  # Based on test_images directory count
    estimated_total_time = avg_time * total_test_images / 3600  # hours

    print(f"Average time per image: {avg_time:.3f} seconds")
    print(f"Total test images: {total_test_images}")
    print(f"Estimated total time: {estimated_total_time:.2f} hours")

    if estimated_total_time > 3.5:
        print("WARNING: Estimated time > 3.5 hours. Consider reducing ensemble or TTA.")
        if estimated_total_time > 4.0:
            print("CRITICAL: Will exceed 4-hour limit. Must optimize.")
    else:
        print("✓ Estimated time within Kaggle limits.")

    return avg_time, estimated_total_time

if __name__ == "__main__":
    measure_inference_time()
