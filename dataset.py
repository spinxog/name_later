import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

class ForgeryDataset(Dataset):
    def __init__(self, image_dir=None, mask_dir=None, transform=None, is_train=True, synthetic=False, image_paths=None, mask_paths=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.is_train = is_train
        self.synthetic = synthetic

        if image_paths is not None and mask_paths is not None:
            # Use provided paths (for k-fold subsets)
            self.image_paths = image_paths
            self.mask_paths = mask_paths
        elif not synthetic:
            self.image_paths = []
            self.mask_paths = []

            if mask_dir:
                # Collect all images from authentic and forged directories
                authentic_dir = os.path.join(image_dir, 'authentic')
                forged_dir = os.path.join(image_dir, 'forged')
                
                if os.path.exists(authentic_dir):
                    for img_file in os.listdir(authentic_dir):
                        if img_file.endswith('.png'):
                            image_id = img_file[:-4]
                            image_path = os.path.join(authentic_dir, img_file)
                            mask_path = os.path.join(mask_dir, f'{image_id}.npy') if os.path.exists(os.path.join(mask_dir, f'{image_id}.npy')) else None
                            self.image_paths.append(image_path)
                            self.mask_paths.append(mask_path)
                
                if os.path.exists(forged_dir):
                    for img_file in os.listdir(forged_dir):
                        if img_file.endswith('.png'):
                            image_id = img_file[:-4]
                            image_path = os.path.join(forged_dir, img_file)
                            mask_path = os.path.join(mask_dir, f'{image_id}.npy') if os.path.exists(os.path.join(mask_dir, f'{image_id}.npy')) else None
                            self.image_paths.append(image_path)
                            self.mask_paths.append(mask_path)
            else:
                # Test data
                for img_file in os.listdir(image_dir):
                    if img_file.endswith('.png'):
                        self.image_paths.append(os.path.join(image_dir, img_file))
                        self.mask_paths.append(None)
        else:
            # Synthetic data will be generated on the fly
            self.image_paths = None
            self.mask_paths = None

    def _find_image_path(self, image_id):
        # Check authentic first
        auth_path = os.path.join(self.image_dir, 'authentic', f'{image_id}.png')
        if os.path.exists(auth_path):
            return auth_path
        # Then forged
        forged_path = os.path.join(self.image_dir, 'forged', f'{image_id}.png')
        if os.path.exists(forged_path):
            return forged_path
        return None

    def __len__(self):
        if self.synthetic:
            return 50000  # ~50k synthetic
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.synthetic:
            return self._generate_synthetic_sample()
        else:
            image_path = self.image_paths[idx]
            try:
                image = cv2.imread(image_path)
                if image is None or image.shape[0] == 0 or image.shape[1] == 0:
                    raise ValueError(f"Failed to load image or invalid shape: {image_path}, shape: {image.shape if image is not None else 'None'}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if image.shape[0] == 0 or image.shape[1] == 0:
                    raise ValueError(f"Invalid image shape after cvtColor: {image.shape}")
            except Exception as e:
                print(f"Warning: Failed to load image {image_path}: {e}. Generating synthetic sample instead.")
                return self._generate_synthetic_sample()

            mask = None
            if self.mask_paths[idx]:
                try:
                    mask = np.load(self.mask_paths[idx])
                    mask = mask.squeeze()  # Squeeze all singleton dimensions
                    if mask.ndim > 2:
                        mask = mask[..., 0]  # Take first channel if multi-channel
                    mask = (mask > 0).astype(np.uint8)
                except Exception as e:
                    print(f"Failed to load mask {self.mask_paths[idx]}: {e}, using zero mask")
                    mask = np.zeros(image.shape[:2], dtype=np.uint8)

                # Check if image or mask shape is invalid
                if image.shape[0] == 0 or image.shape[1] == 0 or mask.shape[0] == 0 or mask.shape[1] == 0:
                    mask = np.zeros(image.shape[:2], dtype=np.uint8)
                # Resize mask to match image size if necessary
                elif mask.shape != image.shape[:2]:
                    try:
                        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                    except cv2.error as e:
                        print(f"cv2.resize failed for mask: {e}, using zero mask")
                        mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    except Exception as e:
                        print(f"Unexpected error resizing mask: {e}, using zero mask")
                        mask = np.zeros(image.shape[:2], dtype=np.uint8)

            if self.transform:
                if mask is not None:
                    transformed = self.transform(image=image, mask=mask)
                    image = transformed['image']
                    mask = transformed['mask']
                else:
                    transformed = self.transform(image=image)
                    image = transformed['image']

            if mask is not None:
                if isinstance(mask, torch.Tensor):
                    mask = mask.float().unsqueeze(0).contiguous()
                else:
                    mask = torch.tensor(mask.astype(np.float32)).unsqueeze(0).contiguous()
            else:
                mask = torch.zeros_like(image[:1]).contiguous()  # For test, dummy mask

            return image, mask

    def _generate_synthetic_sample(self):
        # Load a random authentic image as base
        authentic_dir = os.path.join('recodai-luc-scientific-image-forgery-detection', 'train_images', 'authentic')
        if os.path.exists(authentic_dir):
            authentic_images = [f for f in os.listdir(authentic_dir) if f.endswith('.png')]
        else:
            authentic_images = []

        if not authentic_images:
            # Fallback to random
            image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        else:
            auth_img_path = os.path.join(authentic_dir, random.choice(authentic_images))
            image = cv2.imread(auth_img_path)
            if image is None:
                # Fallback if image loading fails
                image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # Select a patch from another authentic image
        if authentic_images:
            patch_img_path = os.path.join(authentic_dir, random.choice(authentic_images))
            patch_img = cv2.imread(patch_img_path)
            if patch_img is not None:
                patch_img = cv2.cvtColor(patch_img, cv2.COLOR_BGR2RGB)
            else:
                patch_img = None

            # Random patch size and position
            if patch_img is not None:
                patch_h, patch_w = random.randint(64, 256), random.randint(64, 256)
                patch_y = random.randint(0, patch_img.shape[0] - patch_h)
                patch_x = random.randint(0, patch_img.shape[1] - patch_w)
                patch = patch_img[patch_y:patch_y+patch_h, patch_x:patch_x+patch_w]
            else:
                # Skip forgery if patch image failed to load
                patch = None

            # Paste location and apply forgery
            if patch is not None:
                paste_y = random.randint(0, image.shape[0] - patch_h)
                paste_x = random.randint(0, image.shape[1] - patch_w)

                # Apply augmentations: scale, rotate, contrast
                scale = random.uniform(0.8, 1.2)
                angle = random.uniform(-15, 15)
                contrast = random.uniform(0.9, 1.1)

                # Scale
                patch = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                patch_h, patch_w = patch.shape[:2]

                # Rotate
                M = cv2.getRotationMatrix2D((patch_w/2, patch_h/2), angle, 1)
                patch = cv2.warpAffine(patch, M, (patch_w, patch_h))

                # Contrast
                patch = np.clip(patch * contrast, 0, 255).astype(np.uint8)

                # Blend with Poisson or simple alpha
                alpha = random.uniform(0.7, 1.0)
                image[paste_y:paste_y+patch_h, paste_x:paste_x+patch_w] = (
                    alpha * patch + (1 - alpha) * image[paste_y:paste_y+patch_h, paste_x:paste_x+patch_w]
                ).astype(np.uint8)

                # Add noise
                noise = np.random.normal(0, 5, patch.shape).astype(np.uint8)
                image[paste_y:paste_y+patch_h, paste_x:paste_x+patch_w] = np.clip(
                    image[paste_y:paste_y+patch_h, paste_x:paste_x+patch_w] + noise, 0, 255
                )

                # Set mask
                mask[paste_y:paste_y+patch_h, paste_x:paste_x+patch_w] = 1

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        mask = mask.detach().clone().float().unsqueeze(0)
        return image, mask

class BalancedDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        # Load exact, resampled, authentic datasets
        self.exact_dataset = ForgeryDataset(image_dir='data/validation_exact/images', mask_dir='data/validation_exact/masks', transform=transform, is_train=True)
        self.resampled_dataset = ForgeryDataset(image_dir='data/validation_resampled/images', mask_dir='data/validation_resampled/masks', transform=transform, is_train=True)
        authentic_dir = 'recodai-luc-scientific-image-forgery-detection/train_images/authentic'
        authentic_images = [os.path.join(authentic_dir, f) for f in os.listdir(authentic_dir) if f.endswith('.png')]
        self.authentic_dataset = ForgeryDataset(image_paths=authentic_images, mask_paths=[None]*len(authentic_images), transform=transform, is_train=True)

        self.datasets = [self.exact_dataset, self.resampled_dataset, self.authentic_dataset]
        self.ratios = [0.33, 0.33, 0.34]  # exact, resampled, authentic
        self.cum_ratios = np.cumsum(self.ratios)

    def __len__(self):
        return 10000  # Large enough for training

    def __getitem__(self, idx):
        r = np.random.rand()
        dataset_idx = np.searchsorted(self.cum_ratios, r)
        dataset = self.datasets[dataset_idx]
        sample_idx = np.random.randint(len(dataset))
        return dataset[sample_idx]

# Augmentations
def get_train_transforms():
    return A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=20, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], is_check_shapes=False)

def get_val_transforms():
    return A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], is_check_shapes=False)
