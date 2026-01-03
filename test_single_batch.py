import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ForgeryDataset
from models.model import ForgeryDetectionModel
from train import get_transforms, bce_dice_loss

def test_single_batch():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load a small dataset
    dataset = ForgeryDataset('recodai-luc-scientific-image-forgery-detection/train_images',
                             'recodai-luc-scientific-image-forgery-detection/train_masks',
                             transform=get_transforms(False), is_train=True)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    # Model
    model = ForgeryDetectionModel(head_type='correlation_transformer').to(device)

    # Get one batch
    images, masks = next(iter(dataloader))
    images, masks = images.to(device), masks.to(device)

    print("Batch shapes:")
    print(f"Images: {images.shape}, Masks: {masks.shape}")
    print(f"Images dtype: {images.dtype}, Masks dtype: {masks.dtype}")

    # Forward pass in eval mode for stats
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        print(f"Outputs shape: {outputs.shape}")
        print(f"Outputs dtype: {outputs.dtype}")
        print(f"Outputs min: {outputs.min().item():.6f}, max: {outputs.max().item():.6f}, mean: {outputs.mean().item():.6f}")

        probs = torch.sigmoid(outputs)
        print(f"Probs min: {probs.min().item():.6f}, max: {probs.max().item():.6f}, mean: {probs.mean().item():.6f}")

        print(f"Masks min: {masks.min().item():.6f}, max: {masks.max().item():.6f}, mean: {masks.mean().item():.6f}")

    # Loss in train mode for gradient check
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    outputs_train = model(images)  # Forward in train mode
    loss = bce_dice_loss(outputs_train, masks)
    print(f"Raw loss tensor: {loss}")
    print(f"Loss item: {loss.item():.10f}")

    # Check if finite
    print(f"Loss is finite: {torch.isfinite(loss).all().item()}")

    loss.backward()

    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            if len(grad_norms) < 5:  # Print first few
                print(f"Grad norm for {name}: {grad_norm:.6f}")

    print(f"Total params with grads: {len(grad_norms)}")
    print(f"Mean grad norm: {sum(grad_norms)/len(grad_norms) if grad_norms else 0:.6f}")

    # Optimizer step effect
    param_before = list(model.parameters())[0].clone()
    optimizer.step()
    param_after = list(model.parameters())[0]
    norm_diff = (param_before - param_after).norm().item()
    print(f"Param change norm: {norm_diff:.10f}")

if __name__ == "__main__":
    test_single_batch()
