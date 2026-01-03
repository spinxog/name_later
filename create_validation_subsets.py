import os
import numpy as np
import cv2
import torch
from tqdm import tqdm

# Create Exact and Resampled validation subsets (≥500 images each)

def create_exact_copy_subset(n_samples=500):
    """Exact pixel copies: no resampling, no blur, same JPEG, no noise."""
    output_dir = 'data/validation_exact'
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)

    authentic_dir = 'recodai-luc-scientific-image-forgery-detection/train_images/authentic'
    authentic_images = [f for f in os.listdir(authentic_dir) if f.lower().endswith('.png')]

    generated = 0
    attempts = 0
    max_attempts = n_samples * 5

    while generated < n_samples and attempts < max_attempts:
        attempts += 1
        auth_filename = np.random.choice(authentic_images)
        auth_img_path = os.path.join(authentic_dir, auth_filename)
        image = cv2.imread(auth_img_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Select patch
        patch_h = np.random.randint(64, min(128, image.shape[0]) + 1)
        patch_w = np.random.randint(64, min(128, image.shape[1]) + 1)
        patch_y = np.random.randint(0, image.shape[0] - patch_h + 1)
        patch_x = np.random.randint(0, image.shape[1] - patch_w + 1)
        patch = image[patch_y:patch_y+patch_h, patch_x:patch_x+patch_w]

        # Paste location (different from source)
        paste_y = np.random.randint(0, image.shape[0] - patch_h + 1)
        paste_x = np.random.randint(0, image.shape[1] - patch_w + 1)
        if abs(paste_y - patch_y) < patch_h and abs(paste_x - patch_x) < patch_w:
            continue  # Avoid overlap

        # Exact copy: no modifications
        image_out = image.copy()
        image_out[paste_y:paste_y+patch_h, paste_x:paste_x+patch_w] = patch

        # Mask
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        mask[paste_y:paste_y+patch_h, paste_x:paste_x+patch_w] = 1

        img_filename = f'exact_{generated:05d}.png'
        img_path = os.path.join(output_dir, 'images', img_filename)
        cv2.imwrite(img_path, cv2.cvtColor(image_out, cv2.COLOR_RGB2BGR))

        mask_filename = f'exact_{generated:05d}.npy'
        mask_path = os.path.join(output_dir, 'masks', mask_filename)
        np.save(mask_path, mask)

        generated += 1
        if generated % 50 == 0:
            print(f'Generated {generated}/{n_samples} exact copies')

    print(f'Created {generated} exact copy validation images')

def create_resampled_copy_subset(n_samples=500):
    """Resampled copies: scaling ± (0.8–1.4), rotate ± (10–30°), slight blur, different JPEG."""
    output_dir = 'data/validation_resampled'
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)

    authentic_dir = 'recodai-luc-scientific-image-forgery-detection/train_images/authentic'
    authentic_images = [f for f in os.listdir(authentic_dir) if f.lower().endswith('.png')]

    generated = 0
    attempts = 0
    max_attempts = n_samples * 5

    while generated < n_samples and attempts < max_attempts:
        attempts += 1
        auth_filename = np.random.choice(authentic_images)
        auth_img_path = os.path.join(authentic_dir, auth_filename)
        image = cv2.imread(auth_img_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Select patch
        patch_h = np.random.randint(64, min(128, image.shape[0]) + 1)
        patch_w = np.random.randint(64, min(128, image.shape[1]) + 1)
        patch_y = np.random.randint(0, image.shape[0] - patch_h + 1)
        patch_x = np.random.randint(0, image.shape[1] - patch_w + 1)
        patch = image[patch_y:patch_y+patch_h, patch_x:patch_x+patch_w]

        # Resample: scale and rotate
        scale = np.random.uniform(0.8, 1.4)
        angle = np.random.uniform(-30, 30)
        new_h, new_w = int(round(patch.shape[0] * scale)), int(round(patch.shape[1] * scale))

        if new_h <= 0 or new_w <= 0 or new_h > image.shape[0] or new_w > image.shape[1]:
            continue

        patch_resized = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        M = cv2.getRotationMatrix2D((new_w/2, new_h/2), angle, 1)
        patch_rotated = cv2.warpAffine(patch_resized, M, (new_w, new_h))

        # Slight blur
        if np.random.rand() > 0.5:
            patch_rotated = cv2.GaussianBlur(patch_rotated, (3, 3), 0.5)

        # Paste
        paste_y = np.random.randint(0, image.shape[0] - new_h + 1)
        paste_x = np.random.randint(0, image.shape[1] - new_w + 1)

        image_out = image.copy()
        image_out[paste_y:paste_y+new_h, paste_x:paste_x+new_w] = patch_rotated

        # Different JPEG compression
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), np.random.randint(60, 95)]
        _, encoded_img = cv2.imencode('.jpg', cv2.cvtColor(image_out, cv2.COLOR_RGB2BGR), encode_param)
        image_out = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
        image_out = cv2.cvtColor(image_out, cv2.COLOR_BGR2RGB)

        # Mask
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        mask[paste_y:paste_y+new_h, paste_x:paste_x+new_w] = 1

        img_filename = f'resampled_{generated:05d}.png'
        img_path = os.path.join(output_dir, 'images', img_filename)
        cv2.imwrite(img_path, cv2.cvtColor(image_out, cv2.COLOR_RGB2BGR))

        mask_filename = f'resampled_{generated:05d}.npy'
        mask_path = os.path.join(output_dir, 'masks', mask_filename)
        np.save(mask_path, mask)

        generated += 1
        if generated % 50 == 0:
            print(f'Generated {generated}/{n_samples} resampled copies')

    print(f'Created {generated} resampled copy validation images')

if __name__ == "__main__":
    create_exact_copy_subset()
    create_resampled_copy_subset()
