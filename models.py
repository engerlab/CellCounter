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


def check_accuracy(loader, model, device=None):
    '''
    Give the accuracy of the model based on pixel to pixel comparison
    Save the pred images
    Args:
        loader(torch.utils.data.DataLoader): the dataloader of the data, initialized)
        model (nn.Module): an imported NN model
        device(torch.device): the host device
    Return:
        The accuracy of the model across all of the dataset imported by the loader
    '''
    counter=0
    num_batches=0
    accuracy = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Testing'):
            img, mask = batch[0].to(device), batch[1].to(device)
            mask = torch.round(mask, decimals=0)
            
            logits = model(img)
            prob = torch.sigmoid(logits)
            pred = torch.round(prob, decimals=0)
            # for i in range(0, 452):
            #     for j in range(0, 452):
            #         print("mask", mask[0][0][i][j])
            #         print("pred", pred[0][0][i][j])
            correct = (pred == mask).sum() # the total pixels that are correct
            pixels = pred.size(0) * 452 * 452
            accuracy.append(correct / pixels)

            fig, axes = plt.subplots(16, 3, figsize=(15, 30))
            for i, (img, mask, pred) in enumerate(zip(img, mask, pred)):
                axes[i, 0].imshow(img[0].detach().cpu(), cmap='gray')
                axes[i, 0].set_title('Image')
                # axes[i, 1].imshow(img[0].detach().cpu(), cmap='gray')
                axes[i, 1].imshow(mask[0].detach().cpu(), cmap='gray', vmin=0, vmax=1, interpolation='nearest', alpha=0.5)
                axes[i, 1].set_title('Ground truth segmentation')
                # axes[i, 2].imshow(img[0].detach().cpu(), cmap='gray')
                axes[i, 2].imshow(pred[0].detach().cpu(), cmap='gray', vmin=0, vmax=1, interpolation='nearest', alpha=0.5)
                axes[i, 2].set_title('Predicted segmentation')
            
            # Saving the output
            fig.savefig("./preds/prediction" + str(counter) + ".png")
            counter+=1

            num_batches+=1
    
    total_accuracy = sum(accuracy) / num_batches * 100
    return total_accuracy
