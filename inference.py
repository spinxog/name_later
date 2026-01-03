import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from dataset import ForgeryDataset, get_val_transforms
from models.model import ForgeryDetectionModel
from utils import rle_encode, postprocess_mask
import tqdm

def predict_test(model, test_loader, device, threshold=0.5, postprocess=True):
    model.eval()
    predictions = []

    with torch.no_grad():
        for images, _ in tqdm.tqdm(test_loader, desc='Predicting'):
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy()

            for i in range(len(preds)):
                pred = preds[i, 0]
                if postprocess:
                    pred = postprocess_mask(pred)
                pred_bin = (pred > threshold).astype(np.uint8)
                rle = rle_encode(pred_bin)
                predictions.append(rle)

    return predictions

def create_submission(predictions, test_ids, output_path='submission.csv'):
    submission = pd.DataFrame({
        'id': test_ids,
        'rle_mask': predictions
    })
    submission.to_csv(output_path, index=False)
    print(f'Submission saved to {output_path}')

if __name__ == "__main__":
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ForgeryDetectionModel(head_type='correlation_transformer').to(device)
    # Load checkpoint
    checkpoint_path = 'best_model.pth'  # Adjust path
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f'Loaded model from {checkpoint_path}')
    else:
        print('No checkpoint found, using untrained model')

    # Load test data
    test_dir = 'recodai-luc-scientific-image-forgery-detection/test_images'
    test_dataset = ForgeryDataset(test_dir, transform=get_val_transforms(), is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    # Predict
    predictions = predict_test(model, test_loader, device)

    # Get test IDs
    test_ids = [os.path.splitext(os.path.basename(p))[0] for p in test_dataset.image_paths]

    # Create submission
    create_submission(predictions, test_ids)
