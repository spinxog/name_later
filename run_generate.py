import os
import json
import random
import numpy as np
import cv2
from tqdm import tqdm

def generate_synthetic_dataset(image_dir, output_dir, num_samples=50000, seed=42):
    """
    Generate synthetic forgery dataset by pre-computing and saving images and masks.
    Saves metadata for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Create output directories
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)

    # Get list of authentic images
    authentic_dir = os.path.join(image_dir, 'authentic')
    authentic_images = [f for f in os.listdir(authentic_dir) if f.endswith('.png')]
    if not authentic_images:
        raise ValueError("No authentic images found in {}".format(authentic_dir))

    metadata = []

    for idx in tqdm(range(num_samples), desc="Generating synthetic samples"):
        # Load base image
        auth_img_path = os.path.join(authentic_dir, random.choice(authentic_images))
        image = cv2.imread(auth_img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # Select patch
        patch_img_path = os.path.join(authentic_dir, random.choice(authentic_images))
        patch_img = cv2.imread(patch_img_path)
        patch_img = cv2.cvtColor(patch_img, cv2.COLOR_BGR2RGB)

        # Random params
        patch_h = random.randint(64, 256)
        patch_w = random.randint(64, 256)
        patch_y = random.randint(0, patch_img.shape[0] - patch_h)
        patch_x = random.randint(0, patch_img.shape[1] - patch_w)
        patch = patch_img[patch_y:patch_y+patch_h, patch_x:patch_x+patch_w]

        paste_y = random.randint(0, image.shape[0] - patch_h)
        paste_x = random.randint(0, image.shape[1] - patch_w)

        scale = random.uniform(0.8, 1.2)
        angle = random.uniform(-15, 15)
        contrast = random.uniform(0.9, 1.1)
        alpha = random.uniform(0.7, 1.0)
        noise_std = 5

        # Apply scale
        new_h, new_w = int(patch_h * scale), int(patch_w * scale)
        patch = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Apply rotate
        M = cv2.getRotationMatrix2D((new_w/2, new_h/2), angle, 1)
        patch = cv2.warpAffine(patch, M, (new_w, new_h))

        # Adjust paste if scaled/rotated exceeds
        if paste_y + new_h > image.shape[0]:
            new_h = image.shape[0] - paste_y
            patch = patch[:new_h]
        if paste_x + new_w > image.shape[1]:
            new_w = image.shape[1] - paste_x
            patch = patch[:, :new_w]

        # Apply contrast
        patch = np.clip(patch * contrast, 0, 255).astype(np.uint8)

        # Blend
        roi = image[paste_y:paste_y+new_h, paste_x:paste_x+new_w]
        blended = (alpha * patch + (1 - alpha) * roi).astype(np.uint8)
        image[paste_y:paste_y+new_h, paste_x:paste_x+new_w] = blended

        # Add noise
        noise = np.random.normal(0, noise_std, blended.shape).astype(np.int16)
        noisy = np.clip(blended.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        image[paste_y:paste_y+new_h, paste_x:paste_x+new_w] = noisy

        # Set mask
        mask[paste_y:paste_y+new_h, paste_x:paste_x+new_w] = 1

        # Save image
        img_filename = f"synthetic_{idx:05d}.png"
        img_path = os.path.join(output_dir, 'images', img_filename)
        cv2.imwrite(img_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # Save mask
        mask_filename = f"synthetic_{idx:05d}.npy"
        mask_path = os.path.join(output_dir, 'masks', mask_filename)
        np.save(mask_path, mask)

        # Metadata
        meta = {
            'idx': idx,
            'base_image': os.path.basename(auth_img_path),
            'patch_image': os.path.basename(patch_img_path),
            'patch_h': patch_h,
            'patch_w': patch_w,
            'patch_y': patch_y,
            'patch_x': patch_x,
            'paste_y': paste_y,
            'paste_x': paste_x,
            'scale': scale,
            'angle': angle,
            'contrast': contrast,
            'alpha': alpha,
            'noise_std': noise_std,
            'final_patch_h': new_h,
            'final_patch_w': new_w
        }
        metadata.append(meta)

    # Save metadata
    meta_path = os.path.join(output_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Generated {num_samples} synthetic samples in {output_dir}")
    print(f"Metadata saved to {meta_path}")

if __name__ == "__main__":
    image_dir = 'recodai-luc-scientific-image-forgery-detection/train_images'
    output_dir = 'data/synthetic'
    generate_synthetic_dataset(image_dir, output_dir, num_samples=50000)
