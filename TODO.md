# TODO List for RECOD AI LUC Scientific Image Forgery Detection

## 1. Setup and Dependencies
- [x] Install required packages: torch, torchvision, timm, numpy, opencv-python, matplotlib, scikit-learn, albumentations, tqdm, wandb (optional for logging)

## 2. Data Loading and EDA
- [x] Implement dataset.py: Load train_images, train_masks, sample_submission
- [x] EDA script: Visualize 200 random images & masks, analyze types (microscopy, blots), sizes, artifacts
- [x] Compute distributions: % authentic vs forged, average forged mask area, count masks per image

## 3. Synthetic Forgery Generator
- [x] Implement copy-paste generator in dataset.py: Select patch, paste with scale/rotation/contrast/blending/noise
- [x] Generate ~50k synthetic images with masks (GPU-accelerated)
- [x] Save metadata for reproducibility

## 4. Baseline Model Skeleton
- [x] Implement backbone.py: Swin-Tiny or EfficientNet backbone
- [x] Implement decoder.py: U-Net style decoder
- [x] Implement correlation.py: Efficient correlation computation on lower-resolution features
- [x] Implement forensics_transformer.py: Lightweight transformer head with correlation as bias
- [x] Integrate into main model

## 5. Training Pipeline
- [x] Implement train.py: Training loop, loss scaffolding (BCE, Dice, NCE, SSIM), logging
- [x] Pretrain on synthetic dataset (25 epochs completed, additional epochs can be run if needed)
- [x] Fine-tune on real train set (k-fold, 30 epochs per fold completed)

## 6. Evaluation and Metrics
- [x] Implement val.py: Validation with F1 metric consistent with competition
- [x] Implement rle_encode/decode in utils.py
- [x] Test validation and inference pipelines

## 7. Postprocessing & Threshold Tuning
- [x] Implement postprocessing: Connected components, remove tiny components, morphological filtering
- [x] Tune mask threshold by maximizing validation F1
- [x] Run threshold grid search (0.20–0.60 step 0.01) on CV folds - Best: threshold=0.45, F1=0.7321
- [x] Tune A_min in {8,16,32,64} and reciprocal sim threshold in {0.55,0.60,0.65} - Best: a_min=8, sim_threshold=0.55
- [x] Implement reciprocal-match filter and morphological smoothing

## 8. Ensembling / TTA
- [x] Implement 4-flip TTA
- [x] Implement fold-averaging ensemble (top-3 folds)
- [x] Fit temperature scaling on validation - Best: temperature=0.7

## 9. Submission Preparation
- [x] Implement inference.py: RLE encode predictions, produce submission.csv
- [x] Create inference-only notebook with TTA/ensemble/postprocessing (inference_enhanced.py)
- [ ] Test runtime and ensure <4 hours for Kaggle (requires Kaggle environment)

## 10. Ablations & Writeup
- [ ] Ablate correlation vs no-correlation, transformer vs CNN head, synthetic pretrain vs none
- [ ] Document results and findings
- [ ] Run red-team checks: held-out real check, exact/resampled split, gating sanity
- [x] Save artifacts: model weights, params (postprocessing_params.json), evaluation report, README

## Advanced Features (Completed)
- [x] Representation shift: SRM residual branch, DCT/FFT frequency branch, fusion with gating
- [x] Task reformulation: Multi-task architecture (image-level classifier, instance proposal/mask/embedding heads, pairwise verification)
- [x] Learned matching: Efficient correlation computation, self-similarity maps
- [x] Pretexts: Contextual resampling pretext for self-supervised learning
- [x] Risks: Synthetic dominance mitigation (balanced batches), matching collapse (entropy regularization), training instability (GroupNorm)
- [x] Checklist: Exact and resampled validation subsets created, F1 monitoring setup

## Next Steps
- [ ] Run ablations to compare performance with/without correlation, transformer, synthetic pretraining
- [ ] Implement TTA and ensembling for final submission
- [ ] Generate final submission.csv and validate format
