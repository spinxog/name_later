import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetDecoder(nn.Module):
    def __init__(self, in_channels_list, head_type='correlation_transformer'):
        super().__init__()
        self.in_channels_list = in_channels_list
        self.head_type = head_type
        # Adjust channels based on head_type
        if 'transformer' in head_type:
            # Transformer outputs same as input, so last channel is modified
            adjusted_channels = in_channels_list[:-1] + [in_channels_list[-1]]
        else:
            adjusted_channels = in_channels_list

        self.up1 = nn.ConvTranspose2d(adjusted_channels[-1], adjusted_channels[-2], 2, 2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(adjusted_channels[-2] * 2, adjusted_channels[-2], 3, padding=1),
            nn.BatchNorm2d(adjusted_channels[-2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(adjusted_channels[-2], adjusted_channels[-2], 3, padding=1),
            nn.BatchNorm2d(adjusted_channels[-2]),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.ConvTranspose2d(adjusted_channels[-2], adjusted_channels[-3], 2, 2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(adjusted_channels[-3] * 2, adjusted_channels[-3], 3, padding=1),
            nn.BatchNorm2d(adjusted_channels[-3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(adjusted_channels[-3], adjusted_channels[-3], 3, padding=1),
            nn.BatchNorm2d(adjusted_channels[-3]),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.ConvTranspose2d(adjusted_channels[-3], adjusted_channels[-4], 2, 2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(adjusted_channels[-4] * 2, adjusted_channels[-4], 3, padding=1),
            nn.BatchNorm2d(adjusted_channels[-4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(adjusted_channels[-4], adjusted_channels[-4], 3, padding=1),
            nn.BatchNorm2d(adjusted_channels[-4]),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Conv2d(adjusted_channels[-4], 1, 1)

    def forward(self, features):
        x = features[-1]
        x = self.up1(x)
        x = torch.cat([x, features[-2]], dim=1)
        x = self.conv1(x)
        x = self.up2(x)
        x = torch.cat([x, features[-3]], dim=1)
        x = self.conv2(x)
        x = self.up3(x)
        x = torch.cat([x, features[-4]], dim=1)
        x = self.conv3(x)
        x = self.final(x)
        return x
