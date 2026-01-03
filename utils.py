import numpy as np
import torch
from sklearn.metrics import f1_score

def rle_encode(mask):
    """Run-length encoding for binary mask."""
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(rle, shape):
    """Decode RLE to binary mask."""
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

def compute_f1(preds, targets, threshold=0.5):
    """Compute F1 score for binary segmentation."""
    preds_bin = (preds > threshold).astype(np.uint8)
    targets_bin = targets.astype(np.uint8)
    return f1_score(targets_bin.flatten(), preds_bin.flatten())

def postprocess_mask(mask, min_area=10, kernel_size=3):
    """Postprocess mask: remove small components, morphological operations."""
    import cv2
    mask = (mask > 0.5).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            mask[labels == i] = 0
    # Optional: morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

# Loss functions
def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice

def bce_dice_loss(pred, target):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice

def nce_loss(features, temperature=0.1):
    """Noise Contrastive Estimation loss for self-supervised learning."""
    # Simplified NCE for positive pairs
    batch_size = features.shape[0]
    features = torch.nn.functional.normalize(features, dim=1)
    similarity = torch.matmul(features, features.t()) / temperature
    labels = torch.arange(batch_size).to(features.device)
    loss = torch.nn.functional.cross_entropy(similarity, labels)
    return loss

def ssim_loss(pred, target):
    """SSIM loss for image quality."""
    from torchmetrics.functional import structural_similarity_index_measure as ssim
    pred = torch.sigmoid(pred)
    return 1 - ssim(pred, target, data_range=1.0)

def reciprocal_match_filter(mask, features, sim_threshold=0.6, area_ratio_range=(0.2, 5.0)):
    """Filter components based on reciprocal similarity to other components."""
    import cv2
    if mask.sum() == 0:
        return mask

    # Get connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)

    # Extract descriptors for each component
    component_descriptors = []
    component_areas = []
    component_masks = []

    for i in range(1, num_labels):  # Skip background
        component_mask = (labels == i).astype(np.uint8)
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 4:  # Too small to be meaningful
            continue

        # Extract mean feature descriptor (simplified - use spatial average)
        # In practice, you'd use a proper feature extractor
        descriptor = features.mean(axis=(1, 2))  # Global average pooling

        component_descriptors.append(descriptor)
        component_areas.append(area)
        component_masks.append(component_mask)

    if len(component_descriptors) < 2:
        return mask  # Need at least 2 components for matching

    # Compute pairwise similarities
    descriptors = np.array(component_descriptors)
    similarities = np.dot(descriptors, descriptors.T) / (
        np.linalg.norm(descriptors, axis=1, keepdims=True) *
        np.linalg.norm(descriptors, axis=1).reshape(1, -1)
    )

    # Filter components
    filtered_mask = np.zeros_like(mask)
    for i in range(len(component_masks)):
        area_i = component_areas[i]
        has_match = False

        for j in range(len(component_masks)):
            if i == j:
                continue

            area_j = component_areas[j]
            area_ratio = min(area_i, area_j) / max(area_i, area_j)

            if similarities[i, j] >= sim_threshold and area_ratio_range[0] <= area_ratio <= area_ratio_range[1]:
                has_match = True
                break

        if has_match:
            filtered_mask |= component_masks[i]

    return filtered_mask
