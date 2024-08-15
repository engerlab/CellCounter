import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

# Source: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
class UNet(nn.Module):
    def __init__(self, channels):
        super(UNet, self).__init__()

        self.model = nn.Sequential(
            DoubleConv(channels, 64),
            DownSample(64, 128),
            DownSample(128, 256),
            DownSample(256, 512),
            DownSample(512, 1024, single_conv=True),
            UpSample(1024, 512),
            UpSample(512, 256),
            UpSample(256, 128),
            UpSample(128, 64),
            nn.Conv2d(64, 1, (1, 1))
        )
    
    def forward(self, x):
        return self.model(x)



# Double convolution for the UNet
class DoubleConv(nn.Module):
    def __init__(self, input, output, mid=None):
        super().__init__()

        if not mid:
            mid = output
        self.doubleConv = nn.Sequential(
            nn.Conv2d(input, mid, (3,3)),
            nn.ReLU(),
            nn.BatchNorm2d(mid),
            nn.Conv2d(mid, output, (3,3)),
            nn.ReLU(),
            nn.BatchNorm2d(output),
        )

    def forward(self, x):
        return self.doubleConv(x)


# Downsample layers
class DownSample(nn.Module):
    def __init__(self, input, output, single_conv=False):
        super().__init__()
        if single_conv == True:
            self.down = nn.Sequential(
                nn.MaxPool2d((2,2), 2),
                nn.Conv2d(input, output, (3, 3), padding=1)
            )
        else:
            self.down = nn.Sequential(
                nn.MaxPool2d((2,2), 2),
                DoubleConv(input, output)
            )

    def forward(self, x):
        return self.down(x)


# UpSample layers
class UpSample(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DoubleConv(input, output)
        )
    
    def forward(self, x):
        return self.up(x)


# Need to have same size in the output as input
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 16, (5, 5)), # input channel, output channel, kernel siz
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.linearLayers = nn.Sequential(
            nn.Linear(32 * 62 * 62, 48, bias=True),
            nn.ReLU(),
            nn.Linear(48, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, 72, bias=True)
        )
    
    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 32 * 62 * 62)
        x = self.linearLayers(x)

        return x
