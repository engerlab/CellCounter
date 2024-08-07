import os
import wandb
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.model_selection import ShuffleSplit
from dataset import CellDataset, getLabels, getPaths
from models import SimpleNet, UNet


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Hyperparameters
WHICHDATA = 'HCT116' #change
EPOCHS = 20
LR = 0.001
SIZE = (572, 572)
BATCH = 16
MODEL_PATH = './saved/saved_model.pth' # path for saving model
hyperparams_train = {'batch_size': BATCH, #dependent on the image size <-- can resize 300 ish power of 2
               'shuffle': True,
               'num_workers': 0}
hyperparams_test = {'batch_size': BATCH, #dependent on the image size <-- can resize 300 ish power of 2
               'shuffle': False,
               'num_workers': 0}

# The transformation for img and masks of the dataset
transform = nn.Sequential(
    transforms.Resize(SIZE), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # can change according to the dataset
    )
mask_transform = nn.Sequential(
    transforms.Resize((452, 452)), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Grayscale()
    )


def dataset_prep():
    '''
    Train test splitting a dataset and creating data loaders
    Return:
        train_loader (torch.utils.data.dataloader): DataLoader for training dataset
        test_loader (torch.utils.data.dataloader): DataLoaderfor testing dataset
    '''

    # Loading custom dataset: https://stackoverflow.com/questions/51577282/how-do-i-load-custom-image-based-datasets-into-pytorch-for-use-with-a-cnn
    _, img_list = getPaths(dataset='HCT116', img_or_mask='img')
    _, mask_list = getPaths(dataset='HCT116', img_or_mask='mask')


    #train test split
    X = range(0, len(img_list))
    y = getLabels(WHICHDATA)
    samples = {}
    masks = {}
    # labels = {}

    skf = ShuffleSplit(n_splits = 1, train_size=0.8, test_size=0.2)
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        train_data = []
        test_data = []
        train_masks = []
        test_masks = []

        for tridx in train_idx:
            train_data.append(img_list[tridx])
            train_masks.append(mask_list[tridx])
        for teidx in test_idx:
            test_data.append(img_list[teidx])
            test_masks.append(mask_list[teidx])

        samples['train'] = train_data
        samples['test'] = test_data
        masks['train'] = train_masks
        masks['test'] = test_masks


    # Generators
    train_data = CellDataset(samples['train'], masks['train'], transform, mask_transform)
    train_loader = DataLoader(train_data, batch_size=BATCH, shuffle=True, num_workers=0) # Contains 628 samples
    test_data = CellDataset(samples['test'], masks['test'], transform, mask_transform)
    test_loader = DataLoader(test_data, **hyperparams_test) # Contains 159 samples

    return train_loader, test_loader


def train_model(model, train_loader, test_loader):
    '''
    Training a model on the dataset
    Args:
        model (nn.Module): an imported NN model
    '''

    # Initialize Logging

    experiment = wandb.init(
        project='Cell Count',
        anonymous='must',
        config={
            'architecture': 'UNet',
            'dataset': 'HCT116',
            'learning_rate': LR,
            'epochs': EPOCHS,
            'batch_size': BATCH
    })

    training_loss = []
    val_loss = []
    val_loss_min = np.Inf

    # specify loss function
    loss_fn = nn.BCEWithLogitsLoss() # NLLLoss() is another one
    # specify optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    # 20 x 787 = 15740 steps
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1} out of {EPOCHS} epochs')

        # Training Loop
        epoch_training_loss = []
        model.train()
        for batch in tqdm(train_loader, desc='Training'): # Obtain one batch of dataset
            img, mask = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()
            logits = model(img)
            logits = torch.argmax(logits, dim=1)
            mask = torch.argmax(mask, dim=1)
            loss = loss_fn(logits.float(), mask.float())

            # print('loss', loss)
            loss.requires_grad = True
            loss.backward()
            optimizer.step()
            epoch_training_loss.append(loss.item())
            
            wandb.log({'Loss vs. Step': loss})


        # Validation Loop
        epoch_val_loss = []
        model.eval()
        for batch in tqdm(test_loader, desc='Validation'): # Obtain one batch of dataset
            img, mask = batch[0].to(device), batch[1].to(device)
            
            with torch.no_grad():
                logits = model(img)
                logits = torch.argmax(logits, dim=1)
                mask = torch.argmax(mask, dim=1)
                loss = loss_fn(logits.float(), mask.float())
                epoch_val_loss.append(loss.item()) # extract loss value as a Python float

        # Logging epoch metrics
        epoch_training_loss = torch.mean(torch.tensor(epoch_training_loss))
        epoch_val_loss = torch.mean(torch.tensor(epoch_val_loss))
        training_loss.append(epoch_training_loss.item())
        val_loss.append(epoch_val_loss.item())


        # Saving model
        if val_loss_min is None or val_loss_min > epoch_val_loss:
            val_loss_min = epoch_val_loss
            print('val_loss_min is currently:', val_loss_min)
            torch.save(model.state_dict(), MODEL_PATH)
            print("Saved best model!")

        wandb.log({'Loss vs. Epoch': val_loss_min})


if __name__ == '__main__':
    train_loader, test_loader = dataset_prep() 
    model = UNet(channels=3).to(device)
    train_model(model, train_loader, test_loader)
