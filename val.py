import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from utils import compute_f1, postprocess_mask
import tqdm

def validate(model, val_loader, device, threshold=0.5, postprocess=True):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, masks in tqdm.tqdm(val_loader, desc='Validating'):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy()
            targets = masks.cpu().numpy()

            for i in range(len(preds)):
                pred = preds[i, 0]
                target = targets[i, 0]

                # Resize pred to match target size
                pred_resized = torch.nn.functional.interpolate(
                    torch.tensor(pred).unsqueeze(0).unsqueeze(0), 
                    size=target.shape, 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze().numpy()

                if postprocess:
                    pred_resized = postprocess_mask(pred_resized)

                all_preds.append(pred_resized.flatten())
                all_targets.append(target.flatten())

    # Compute F1 at pixel level
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    f1 = f1_score(all_targets, (all_preds > threshold).astype(int))

    return f1

def find_best_threshold(model, val_loader, device, thresholds=np.arange(0.1, 0.9, 0.1)):
    best_f1 = 0
    best_thresh = 0.5

    for thresh in thresholds:
        f1 = validate(model, val_loader, device, threshold=thresh, postprocess=False)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    return best_thresh, best_f1
