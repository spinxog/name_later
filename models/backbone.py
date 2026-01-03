import torch
import torch.nn as nn
import timm

class Backbone(nn.Module):
    def __init__(self, model_name='efficientnet_b0', pretrained=True):
        super().__init__()
        # Use EfficientNet-B0 as default, more stable feature extraction
        self.encoder = timm.create_model(model_name, pretrained=pretrained, features_only=True, out_indices=[1,2,3,4])
        self.out_channels = self.encoder.feature_info.channels()

    def forward(self, x):
        features = self.encoder(x)
        return features  # List of feature maps at different scales

# Alternative Swin Transformer (if needed)
class SwinBackbone(nn.Module):
    def __init__(self, model_name='swin_tiny_patch4_window7_224', pretrained=True):
        super().__init__()
        self.encoder = timm.create_model(model_name, pretrained=pretrained, features_only=True, out_indices=[0,1,2,3])  # Adjusted indices

    def forward(self, x):
        features = self.encoder(x)
        return features
