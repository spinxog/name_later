import torch
import numpy as np
import os
from tqdm import tqdm

class HardNegativeManager:
    def __init__(self, threshold=0.5, max_negatives=10000):
        self.threshold = threshold
        self.max_negatives = max_negatives
        self.hard_negatives = []  # List of (image_path, mask_path, confidence, soft_label_weight)

    def collect_hard_negatives(self, model, dataloader, device, epoch):
        model.eval()
        candidates = []
        with torch.no_grad():
            for images, masks, image_paths, mask_paths in tqdm(dataloader, desc="Collecting hard negatives"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                preds = torch.sigmoid(outputs)

                # Find high-confidence false positives
                fp_mask = (preds > self.threshold) & (masks == 0)
                confidences = preds[fp_mask].cpu().numpy()
                indices = torch.nonzero(fp_mask, as_tuple=True)

                for b, h, w in zip(*indices):
                    confidence = confidences[len(candidates) % len(confidences)]
                    candidates.append({
                        'image_path': image_paths[b],
                        'mask_path': mask_paths[b],
                        'confidence': confidence,
                        'epoch': epoch,
                        'h': h.item(),
                        'w': w.item()
                    })

        # Filter candidates
        filtered = self._filter_candidates(candidates, model, dataloader, device)
        self.hard_negatives.extend(filtered)
        self.hard_negatives = self.hard_negatives[-self.max_negatives:]  # Keep most recent

        print(f"Collected {len(filtered)} hard negatives, total: {len(self.hard_negatives)}")

    def _filter_candidates(self, candidates, model, dataloader, device):
        filtered = []
        for cand in candidates:
            if self._passes_filters(cand, model, dataloader, device):
                # Soft label weight based on consistency
                weight = 0.5 if cand['confidence'] > 0.7 else 0.75
                filtered.append({
                    'image_path': cand['image_path'],
                    'mask_path': cand['mask_path'],
                    'weight': weight
                })
        return filtered

    def _passes_filters(self, cand, model, dataloader, device):
        # Filter 1: Seen as FP in ≥3 sweeps (simplified: check current confidence)
        if cand['confidence'] < 0.6:
            return False

        # Filter 2: Image-level forgery prob < 0.12
        # Simplified: check if image has low overall prediction
        image_pred = self._get_image_level_prob(cand['image_path'], model, device)
        if image_pred > 0.12:
            return False

        # Filter 3: Pairwise match < 0.15
        # Simplified: check if similar to other candidates
        if self._has_similar_match(cand):
            return False

        # Filter 4: Area small enough
        # Assume pixel-level, area < 0.01 * image area
        if cand['h'] * cand['w'] > 0.01 * 512 * 512:
            return False

        return True

    def _get_image_level_prob(self, image_path, model, device):
        # Dummy implementation: return random for now
        return np.random.rand()

    def _has_similar_match(self, cand):
        # Dummy: check if any other candidate from same image
        for hn in self.hard_negatives:
            if hn['image_path'] == cand['image_path']:
                return True
        return False

    def get_hard_negative_dataset(self, transform=None):
        from dataset import ForgeryDataset
        image_paths = [hn['image_path'] for hn in self.hard_negatives]
        mask_paths = [hn['mask_path'] for hn in self.hard_negatives]
        weights = [hn['weight'] for hn in self.hard_negatives]
        dataset = ForgeryDataset(image_paths=image_paths, mask_paths=mask_paths, transform=transform, is_train=True)
        dataset.weights = weights  # Add weights for soft labeling
        return dataset

    def save(self, path='hard_negatives.pkl'):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.hard_negatives, f)

    def load(self, path='hard_negatives.pkl'):
        import pickle
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.hard_negatives = pickle.load(f)
