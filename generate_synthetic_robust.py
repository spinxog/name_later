import os
import numpy as np
import cv2
import torch
from tqdm import tqdm

# Generate ~50k synthetic images with GPU acceleration where possible
image_dir = 'recodai-luc-scientific-image-forgery-detection/train_images'
output_dir = 'data/synthetic'
os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)

authentic_dir = os.path.join(image_dir, 'authentic')
authentic_images = [f for f in os.listdir(authentic_dir) if f.lower().endswith('.png')]

n_samples = 50000
generated = 0
attempts = 0
max_attempts = n_samples * 5  # avoid infinite loops if things are bad

while generated < n_samples and attempts < max_attempts:
    attempts += 1
    auth_filename = np.random.choice(authentic_images)
    auth_img_path = os.path.join(authentic_dir, auth_filename)
    image = cv2.imread(auth_img_path)
    if image is None:
        # corrupted / unreadable, skip
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # pick a patch image (can be the same file; that's fine)
    patch_filename = np.random.choice(authentic_images)
    patch_img_path = os.path.join(authentic_dir, patch_filename)
    patch_img = cv2.imread(patch_img_path)
    if patch_img is None:
        continue
    patch_img = cv2.cvtColor(patch_img, cv2.COLOR_BGR2RGB)

    # choose random patch within the patch_img
    patch_h = np.random.randint(64, min(256, patch_img.shape[0]) + 1)
    patch_w = np.random.randint(64, min(256, patch_img.shape[1]) + 1)
    # safe crop coords
    if patch_img.shape[0] - patch_h <= 0 or patch_img.shape[1] - patch_w <= 0:
        # fall back to center crop if random size equals image size
        patch = patch_img[:patch_h, :patch_w]
    else:
        patch_y = np.random.randint(0, patch_img.shape[0] - patch_h + 1)
        patch_x = np.random.randint(0, patch_img.shape[1] - patch_w + 1)
        patch = patch_img[patch_y:patch_y+patch_h, patch_x:patch_x+patch_w]

    # Compute allowed scale so resized patch fits inside the target image
    # Prevent dividing by zero (patch_h/w > 0 because we sampled >=64)
    max_scale_h = image.shape[0] / patch.shape[0]
    max_scale_w = image.shape[1] / patch.shape[1]
    max_allowed_scale = min(max_scale_h, max_scale_w, 1.2)  # keep original upper bound 1.2

    if max_allowed_scale <= 0:
        # patch larger than image even without scaling (very unlikely); skip
        continue

    # choose scale: prefer 0.8..1.2 but clamp to max_allowed_scale
    if max_allowed_scale < 0.8:
        scale = max_allowed_scale  # shrink to fit
    else:
        scale = np.random.uniform(0.8, max_allowed_scale)

    contrast = np.random.uniform(0.9, 1.1)
    alpha = np.random.uniform(0.7, 1.0)
    noise_std = 5

    new_h, new_w = int(round(patch.shape[0] * scale)), int(round(patch.shape[1] * scale))
    # final safety clamp (shouldn't be necessary but safe)
    new_h = min(new_h, image.shape[0])
    new_w = min(new_w, image.shape[1])

    if new_h <= 0 or new_w <= 0:
        continue

    # Use GPU for resizing and blending if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    patch_tensor = torch.tensor(patch, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)  # 1x3xHxW
    patch_resized_tensor = torch.nn.functional.interpolate(patch_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
    patch_resized = (patch_resized_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * contrast).clip(0, 255).astype(np.uint8)

    # choose paste location such that it fully fits in the image
    max_y = image.shape[0] - new_h
    max_x = image.shape[1] - new_w
    if max_y < 0 or max_x < 0:
        # shouldn't happen because we enforced max_allowed_scale, but keep safe
        continue

    paste_y = np.random.randint(0, max_y + 1)
    paste_x = np.random.randint(0, max_x + 1)

    roi = image[paste_y:paste_y+new_h, paste_x:paste_x+new_w]
    blended = (alpha * patch_resized + (1 - alpha) * roi).astype(np.uint8)

    # add gaussian noise
    noise = np.random.normal(0, noise_std, blended.shape).astype(np.int16)
    noisy = np.clip(blended.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    image_out = image.copy()
    image_out[paste_y:paste_y+new_h, paste_x:paste_x+new_w] = noisy

    # create mask
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    mask[paste_y:paste_y+new_h, paste_x:paste_x+new_w] = 1

    img_filename = f'synthetic_{generated:05d}.png'
    img_path = os.path.join(output_dir, 'images', img_filename)
    cv2.imwrite(img_path, cv2.cvtColor(image_out, cv2.COLOR_RGB2BGR))

    mask_filename = f'synthetic_{generated:05d}.npy'
    mask_path = os.path.join(output_dir, 'masks', mask_filename)
    np.save(mask_path, mask)

    generated += 1
    if generated % 10 == 0:
        print(f'Generated {generated}/{n_samples}')

print(f'Generated {generated} synthetic samples (attempted {attempts} tries)')
