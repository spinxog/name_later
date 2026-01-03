import numpy as np
import cv2
import os

mask_dir = 'recodai-luc-scientific-image-forgery-detection/train_masks'
image_dir = 'recodai-luc-scientific-image-forgery-detection/train_images'

count = 0
for mask_file in os.listdir(mask_dir):
    if mask_file.endswith('.npy'):
        mask_path = os.path.join(mask_dir, mask_file)
        mask = np.load(mask_path)
        image_id = mask_file[:-4]
        auth_path = os.path.join(image_dir, 'authentic', f'{image_id}.png')
        if os.path.exists(auth_path):
            image = cv2.imread(auth_path)
        else:
            forged_path = os.path.join(image_dir, 'forged', f'{image_id}.png')
            if os.path.exists(forged_path):
                image = cv2.imread(forged_path)
            else:
                continue
        if image is None:
            continue
        if mask.shape[:2] != image.shape[:2]:
            print(f"Mismatch: Mask {mask_file} shape {mask.shape}, Image shape {image.shape}")
            count += 1
            if count > 10:
                break
if count == 0:
    print("All masks and images have matching shapes.")
else:
    print(f"Found {count} mismatches.")
