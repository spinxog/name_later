import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Subset
from dataset import ForgeryDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from models.model import ForgeryDetectionModel
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm
import os
import torch.nn.functional as F
import glob

def get_transforms(is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(512, 512),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def bce_dice_loss(pred, target):
    bce = nn.BCEWithLogitsLoss()(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice

def nce_loss(features, temperature=0.1):
    # Simplified NCE to avoid OOM: contrast within batch, not all pixels
    B, C, H, W = features.shape
    features = features.view(B, C, -1).permute(0, 2, 1).contiguous()  # [B, H*W, C]
    features = features.view(-1, C)  # [B*H*W, C]
    features = F.normalize(features, dim=1)

    # Sample a subset for contrast
    num_samples = min(1024, features.shape[0])
    indices = torch.randperm(features.shape[0], device=features.device)[:num_samples]
    features = features[indices]

    # Contrast against batch
    logits = torch.matmul(features, features.T) / temperature
    labels = torch.arange(num_samples, device=features.device)
    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss

def combined_loss(pred, target, features=None, gates=None):
    bce_dice = bce_dice_loss(pred, target)
    loss = bce_dice
    if features is not None:
        nce = nce_loss(features)
        loss += 0.1 * nce  # Weight NCE lightly
    if gates is not None:
        # Entropy regularizer for gating: prevent saturation
        entropy = -0.01 * (gates * torch.log(gates + 1e-8) + (1 - gates) * torch.log(1 - gates + 1e-8)).mean()
        loss += entropy
    return loss

def train_epoch(model, dataloader, optimizer, device, use_nce=False):
    model.train()
    total_loss = 0
    for images, masks in tqdm(dataloader, desc="Training"):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        # Resize outputs to match masks if needed
        if outputs.shape[-2:] != masks.shape[-2:]:
            outputs = nn.functional.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)

        if use_nce and hasattr(model, 'backbone'):
            features = model.backbone(images)  # Get features for NCE
            loss = combined_loss(outputs, masks, features[0])  # Use shallow features
        else:
            loss = bce_dice_loss(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def validate_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validating"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = bce_dice_loss(outputs, masks)
            total_loss += loss.item()

            preds = torch.sigmoid(outputs) > 0.5
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(masks.cpu().numpy().flatten())

    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='binary')
    return total_loss / len(dataloader), precision, recall, f1

def pretrain_on_synthetic(model, device, epochs=20):
    print("Pretraining on synthetic data...")
    synthetic_dataset = ForgeryDataset('data/synthetic/images', 'data/synthetic/masks',
                                       transform=get_transforms(True), is_train=True, synthetic=False)
    synthetic_loader = DataLoader(synthetic_dataset, batch_size=2, shuffle=True, num_workers=1)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Check for existing checkpoints (old format or new)
    checkpoint_files = glob.glob('pretrained_model_epoch_*.pth')
    if checkpoint_files:
        epochs_done = [int(f.split('_')[-1].split('.')[0]) for f in checkpoint_files]
        latest_epoch = max(epochs_done)
        checkpoint_path = f'pretrained_model_epoch_{latest_epoch}.pth'
        model.load_state_dict(torch.load(checkpoint_path), strict=False)
        start_epoch = latest_epoch
        print(f"Resumed from epoch {start_epoch}")
        # Adjust scheduler
        optimizer.step()  # Dummy step to initialize scheduler
        for _ in range(start_epoch):
            scheduler.step()
    else:
        start_epoch = 0

    for epoch in range(start_epoch, epochs):
        train_loss = train_epoch(model, synthetic_loader, optimizer, device, use_nce=True)
        scheduler.step()
        print(f"Pretrain Epoch {epoch+1}: Loss: {train_loss:.4f}")

        # Save checkpoint (old format for compatibility)
        torch.save(model.state_dict(), f'pretrained_model_epoch_{epoch+1}.pth')

    torch.save(model.state_dict(), 'pretrained_model.pth')
    print("Saved pretrained model")

def finetune_on_real(model, device, epochs=30, k_folds=5):
    print("Finetuning on real data with k-fold cross-validation...")
    data_dir = 'recodai-luc-scientific-image-forgery-detection'
    full_dataset = ForgeryDataset(os.path.join(data_dir, 'train_images'), os.path.join(data_dir, 'train_masks'),
                                  transform=None, is_train=True)  # No transform here, apply in loaders

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(full_dataset)))):
        print(f"\nFold {fold+1}/{k_folds}")

        # Create subsets
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)

        # Apply transforms via custom dataset wrapper or modify dataset to apply transforms
        train_dataset = ForgeryDataset.__new__(ForgeryDataset)
        train_dataset.__dict__.update(train_subset.dataset.__dict__)
        train_dataset.image_paths = [train_subset.dataset.image_paths[i] for i in train_idx]
        train_dataset.mask_paths = [train_subset.dataset.mask_paths[i] for i in train_idx]
        train_dataset.transform = get_transforms(True)

        val_dataset = ForgeryDataset.__new__(ForgeryDataset)
        val_dataset.__dict__.update(val_subset.dataset.__dict__)
        val_dataset.image_paths = [val_subset.dataset.image_paths[i] for i in val_idx]
        val_dataset.mask_paths = [val_subset.dataset.mask_paths[i] for i in val_idx]
        val_dataset.transform = get_transforms(False)

        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

        # Check for existing checkpoint for this fold
        fold_checkpoint_path = f'finetuned_fold_{fold+1}_checkpoint.pth'
        if os.path.exists(fold_checkpoint_path):
            checkpoint = torch.load(fold_checkpoint_path)
            model.load_state_dict(checkpoint['model'])
            start_epoch = checkpoint['epoch']
            best_f1 = checkpoint['best_f1']
            print(f"Resumed fold {fold+1} from epoch {start_epoch}")
        else:
            # Start from pretrained for each fold
            model.load_state_dict(torch.load('pretrained_model.pth'))
            start_epoch = 0
            best_f1 = 0

        optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Adjust scheduler if resuming
        optimizer.step()  # Dummy step to initialize scheduler
        for _ in range(start_epoch):
            scheduler.step()

        for epoch in range(start_epoch, epochs):
            train_loss = train_epoch(model, train_loader, optimizer, device)
            val_loss, precision, recall, f1 = validate_epoch(model, val_loader, device)
            scheduler.step()

            print(f"Fold {fold+1}, Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

            # Save checkpoint
            torch.save({'model': model.state_dict(), 'epoch': epoch+1, 'best_f1': best_f1}, fold_checkpoint_path)

            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), f'best_finetuned_fold_{fold+1}.pth')

        fold_results.append(best_f1)
        print(f"Fold {fold+1} best F1: {best_f1:.4f}")

    avg_f1 = np.mean(fold_results)
    print(f"Average F1 across folds: {avg_f1:.4f}")

    # Save the best model across folds
    best_fold = np.argmax(fold_results) + 1
    torch.save(torch.load(f'best_finetuned_fold_{best_fold}.pth'), 'best_finetuned_model.pth')
    print(f"Saved best model from fold {best_fold}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model
    model = ForgeryDetectionModel(head_type='correlation_transformer').to(device)

    # Pretrain on synthetic
    pretrain_on_synthetic(model, device, epochs=25)

    # Finetune on real
    finetune_on_real(model, device, epochs=30, k_folds=5)

if __name__ == "__main__":
    main()

