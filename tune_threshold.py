import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from dataset import ForgeryDataset
from train import get_transforms
from models.model import ForgeryDetectionModel
from utils import postprocess_mask, compute_f1
import os
import json

def tune_threshold_on_fold(model_path, val_loader, device, thresholds=np.arange(0.20, 0.61, 0.01),
                          a_mins=[8, 16, 32, 64], sim_thresholds=[0.55, 0.60, 0.65]):
    """Tune threshold, A_min, and sim_threshold on a single fold."""
    model = ForgeryDetectionModel(head_type='correlation_transformer').to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    best_f1 = 0
    best_params = {'threshold': 0.5, 'a_min': 16, 'sim_threshold': 0.6}

    all_preds = []
    all_targets = []
    all_features = []  # For reciprocal matching

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy()
            targets = masks.cpu().numpy()

            # Get features for reciprocal matching (simplified - using backbone features)
            features = model.backbone(images)[0].cpu().numpy()  # Use shallow features

            for i in range(len(preds)):
                pred = preds[i, 0]
                target = targets[i, 0]
                feat = features[i]

                # Resize pred to match target
                pred_resized = torch.nn.functional.interpolate(
                    torch.tensor(pred).unsqueeze(0).unsqueeze(0),
                    size=target.shape,
                    mode='bilinear',
                    align_corners=False
                ).squeeze().numpy()

                all_preds.append(pred_resized)
                all_targets.append(target)
                all_features.append(feat)

    # Grid search
    for threshold in thresholds:
        for a_min in a_mins:
            for sim_thresh in sim_thresholds:
                f1_scores = []
                for pred, target, feat in zip(all_preds, all_targets, all_features):
                    # Postprocess
                    processed = postprocess_mask(pred, threshold=threshold, min_area=a_min)
                    # Note: reciprocal_match_filter not implemented yet, skip for now

                    # Compute F1
                    f1 = compute_f1(processed, target)
                    f1_scores.append(f1)

                avg_f1 = np.mean(f1_scores)
                if avg_f1 > best_f1:
                    best_f1 = avg_f1
                    best_params = {'threshold': threshold, 'a_min': a_min, 'sim_threshold': sim_thresh}

    return best_params, best_f1

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = 'recodai-luc-scientific-image-forgery-detection'

    # Load full dataset
    full_dataset = ForgeryDataset(os.path.join(data_dir, 'train_images'),
                                  os.path.join(data_dir, 'train_masks'),
                                  transform=get_transforms(False))

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_params = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(full_dataset)))):
        print(f"Tuning fold {fold+1}...")

        # Create val subset
        val_subset = torch.utils.data.Subset(full_dataset, val_idx)
        val_loader = DataLoader(val_subset, batch_size=4, shuffle=False, num_workers=4)

        model_path = f'best_finetuned_fold_{fold+1}.pth'
        if os.path.exists(model_path):
            params, f1 = tune_threshold_on_fold(model_path, val_loader, device)
            fold_params.append(params)
            print(f"Fold {fold+1}: Best F1 {f1:.4f}, Params: {params}")
        else:
            print(f"Model for fold {fold+1} not found, skipping")

    # Average parameters across folds
    if fold_params:
        avg_params = {}
        for key in fold_params[0].keys():
            avg_params[key] = np.mean([p[key] for p in fold_params])

        # Save parameters
        with open('postprocessing_params.json', 'w') as f:
            json.dump(avg_params, f, indent=2)

        print(f"Average params across folds: {avg_params}")

if __name__ == "__main__":
    main()
