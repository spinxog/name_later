import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from dataset import ForgeryDataset
import random

def load_sample_data(data_dir='recodai-luc-scientific-image-forgery-detection'):
    train_images_dir = os.path.join(data_dir, 'train_images')
    train_masks_dir = os.path.join(data_dir, 'train_masks')

    dataset = ForgeryDataset(train_images_dir, train_masks_dir, transform=None, is_train=False)

    return dataset

def visualize_samples(dataset, num_samples=5):
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))

    for i in range(num_samples):
        image, mask = dataset[i]
        if hasattr(image, 'numpy'):
            image = image.numpy()
        if hasattr(mask, 'numpy'):
            mask = mask.numpy()

        # Ensure correct shape: CHW to HWC
        if image.shape[0] == 3:  # CHW
            image = image.transpose(1, 2, 0)
        mask = mask.squeeze()  # Squeeze all dimensions
        if mask.ndim > 2:
            mask = mask[0]  # Take first channel if multi-channel

        # Denormalize if needed
        if image.max() <= 1.0:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = std * image + mean
            image = np.clip(image, 0, 1)
        else:
            image = image / 255.0

        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f'Image {i+1}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title(f'Mask {i+1}')
        axes[i, 1].axis('off')

        # Overlay
        overlay = image.copy()
        overlay[mask > 0.5] = [1, 0, 0]  # Red for forgery
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(f'Overlay {i+1}')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()

def analyze_distribution(dataset):
    total_images = len(dataset)
    forged_count = 0
    mask_areas = []
    masks_per_image = []

    for i in range(len(dataset)):
        _, mask = dataset[i]
        mask_np = mask.squeeze().numpy()
        if mask_np.sum() > 0:
            forged_count += 1
            mask_areas.append(mask_np.sum())
            masks_per_image.append(1)  # Assuming one mask per image for now

    authentic_pct = (total_images - forged_count) / total_images * 100
    forged_pct = forged_count / total_images * 100
    avg_mask_area = np.mean(mask_areas) if mask_areas else 0
    avg_masks_per_image = np.mean(masks_per_image) if masks_per_image else 0

    print(f"Total images: {total_images}")
    print(f"Authentic: {authentic_pct:.2f}%")
    print(f"Forged: {forged_pct:.2f}%")
    print(f"Average forged mask area: {avg_mask_area:.2f} pixels")
    print(f"Average masks per image: {avg_masks_per_image:.2f}")

def analyze_image_types(dataset):
    microscopy_count = 0
    blot_count = 0
    other_count = 0

    for i in range(min(100, len(dataset))):  # Sample 100 to avoid issues
        try:
            image, _ = dataset[i]
            if hasattr(image, 'numpy'):
                image_np = image.numpy().transpose(1, 2, 0).astype(np.uint8)
            else:
                image_np = image.transpose(1, 2, 0).astype(np.uint8)

            # Ensure 3 channels
            if image_np.shape[2] != 3:
                continue

            # Simple heuristic: check for circular patterns or high contrast
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

            if edge_density > 0.1:  # High edges, maybe microscopy
                microscopy_count += 1
            elif np.std(gray) > 50:  # High variance, maybe blots
                blot_count += 1
            else:
                other_count += 1
        except Exception as e:
            print(f"Error processing image {i}: {e}")
            continue

    print(f"Microscopy-like: {microscopy_count}")
    print(f"Blot-like: {blot_count}")
    print(f"Other: {other_count}")

if __name__ == "__main__":
    dataset = load_sample_data()
    analyze_distribution(dataset)
    analyze_image_types(dataset)
    visualize_samples(dataset, 20)  # Visualize 20 for quick check
