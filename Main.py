## Utilities
import os
import wandb
import time
from datetime import timedelta
from tqdm import tqdm

## Image Processing
import numpy as np
import monai as mn
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import ShuffleSplit

## Custom Python files
from dataset import CellDataset, getLabels, getPaths
from models import SimpleNet, UNet
from evaluation import validation, check_accuracy
from dice import dice_coeff, dice_loss

# Hyperparameters
lr = 5e-4
WHICHDATA = 'HCT116' #change
EPOCHS = 30
BATCH = 16
SIZE = (572, 572)
MODEL_PATH = './saved/saved_model.pth' # path for saving model

# Hyperparameters
hyperparams_train = {'batch_size': BATCH, #dependent on the image size <-- can resize 300 ish power of 2
               'shuffle': True,
               'num_workers': 0}
hyperparams_test = {'batch_size': BATCH, #dependent on the image size <-- can resize 300 ish power of 2
               'shuffle': False,
               'num_workers': 0}

# The transformation for img and masks of the dataset
train_transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(30),
    # transforms.RandomZoomOut(side_range=(1.0, 4.0), p=0.3),
    transforms.Resize(SIZE),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # can change according to the dataset
    ])
test_transform = transforms.Compose([
    transforms.Resize(SIZE),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # can change according to the dataset
    ])


# Initialize Logging
experiment = wandb.init(
        project='Cell Count',
        anonymous='must',
        config={
            'architecture': 'UNet',
            'dataset': 'HCT116',
            'learning_rate': lr,
            'epochs': EPOCHS,
            'batch_size': BATCH
    })


def dataset_prep():
    '''
    Train test splitting a dataset and creating data loaders
    Returns:
        train_loader (torch.utils.data.dataloader): DataLoader for training dataset
        val_loader (torch.utils.data.dataloader): DataLoader for validation dataset
        test_loader (torch.utils.data.dataloader): DataLoader for testing dataset
    '''

    # Loading custom dataset: https://stackoverflow.com/questions/51577282/how-do-i-load-custom-image-based-datasets-into-pytorch-for-use-with-a-cnn
    _, img_list = getPaths(dataset='HCT116', img_or_mask='img')
    _, mask_list = getPaths(dataset='HCT116', img_or_mask='mask')


    #train test split
    X = range(0, len(img_list))
    y = getLabels(WHICHDATA)
    samples = {}
    masks = {}
    train_indices = []


    ss = ShuffleSplit(n_splits = 1, train_size=0.8, test_size=0.2)
    for train_idx, test_idx in ss.split(X):
        test_data = []
        test_masks = []
        for teidx in test_idx: # Adding testing data into temporary lists
            test_data.append(img_list[teidx])
            test_masks.append(mask_list[teidx])

        # Adding test set to dictionaries
        train_indices = train_idx
        samples['test'] = test_data
        masks['test'] = test_masks
        

    for train_idx, val_idx in ss.split(train_indices): # 80 20 split for training and validation dataset
        train_data = []
        val_data = []
        train_masks = []
        val_masks = []
        for tridx in train_idx: # Adding training data into temporary lists
            train_data.append(img_list[tridx])
            train_masks.append(mask_list[tridx])
        for validx in val_idx: # Adding validation data into temporary lists
            val_data.append(img_list[validx])
            val_masks.append(mask_list[validx])

        # Adding training and validation sets to dictionaries
        samples['train'] = train_data
        samples['validation'] = val_data
        masks['train'] = train_masks
        masks['validation'] = val_masks
        

    # Generators
    train_data = CellDataset(samples['train'], masks['train'], train_transform)
    train_loader = DataLoader(train_data, batch_size=BATCH, shuffle=True, num_workers=0) # Contains 503 samples
    val_data = CellDataset(samples['validation'], masks['validation'], test_transform)
    val_loader = DataLoader(val_data, batch_size=BATCH, shuffle=True, num_workers=0) # Contains 126 samples
    test_data = CellDataset(samples['test'], masks['test'], test_transform)
    test_loader = DataLoader(test_data, **hyperparams_test) # Contains 158 samples


    return train_loader, val_loader, test_loader


def one_hot(target):
    '''
    Convert an image to one hot format
    Arg:
        targets (torch.Tensor): an image in tensor format
    Return:
        one_hot (torch.Tensor): tensor of shape of target that is in one-hot format
    '''  
    target_extend=target.clone().to(torch.int64)
    one_hot = torch.cuda.FloatTensor(target_extend.size(0), 2, target_extend.size(2), target_extend.size(3)).zero_()
    one_hot.scatter_(1, target_extend, 1) 
    return one_hot


def train_model(model, train_loader, val_loader):
    '''
    Training a model on the dataset
    Args:
        model (nn.Module): an imported NN model
        train_loader (torch.utils.data.DataLoader): the dataloader of the training data, initialized
        val_loader (torch.utils.data.DataLoader): the dataloader of the validation data, initialized
    '''

    # Counters
    train_steps=0
    val_steps=0
    epoch_num=0
    count=0

    # Keep track of the loss
    training_loss = []
    val_loss = []
    val_loss_min = np.Inf

    # specify loss function
    loss_fn = nn.BCEWithLogitsLoss() # NLLLoss() is another one
    # specify optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    # specify learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)  # goal: maximize Dice score

    # 20 x 787 = 15740 steps
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1} out of {EPOCHS} epochs')

        # Training Loop
        epoch_training_loss = []
        model.train()
        for batch in tqdm(train_loader, desc='Training'): # Obtain one batch of dataset
            img, mask = batch[0].to(device), batch[1].to(device)
            mask = torch.round(mask, decimals=0)
            # mask = one_hot(mask).to(device)

            optimizer.zero_grad()
            logits = model(img)
            # print(logits[0, :, :, :])
            loss = loss_fn(logits.float(), mask.float())
            loss += dice_loss(torch.sigmoid(logits.float()), mask.float(), multiclass=False)

            loss.backward()
            optimizer.step()
            epoch_training_loss.append(loss.item()) # total loss per batch
            
            train_steps+=1
            wandb.log({'Loss vs. Step': loss})

        # Validation Loop
        val_dice_score, epoch_val_loss = validation(val_loader, loss_fn, model, device, epoch_num)
        scheduler.step(val_dice_score)

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

        wandb.log({'Validation Loss vs. Epoch': val_loss_min})
        wandb.log({"Training Loss vs. Epoch": epoch_training_loss})
        wandb.log({"Dice Score vs. Epoch": val_dice_score}) # Should be maximized
        epoch_num+=1
    #     print("Steps in Train per epoch:", train_steps) # 40
    #     print("Steps in Validation per epoch:", val_steps) # 10
    # print("Actual num of epochs:", epoch_num) #20


def testing_model(test_loader):
    '''
    Testing the pre-trained model on the testing dataset
    Args:
        test_loader (torch.utils.data.DataLoader): the dataloader of the testing data, initialized
    '''
    counter=0

    model = UNet(channels=3)
    model.load_state_dict(torch.load(MODEL_PATH))
    model = model.to(device)
    model.eval()

    accuracy = check_accuracy(test_loader, model, device=device)

    print(f"The accuracy of the testing dataset is {accuracy}%")
    print("\nTesting has finished, please find the results in ./preds")
            

if __name__ == '__main__':
    start = time.time()
    # Send email alert when job begins
    experiment.alert(
        title='Main.py Has Begun Running',
        text='Start')

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    # Main section
    train_loader, val_loader, test_loader = dataset_prep()
    model = UNet(channels=3).to(device)
    train_model(model, train_loader, val_loader)
    testing_model(test_loader)

    end = time.time()
    elapsed_time = str(timedelta(seconds=(end-start)))
    # Send email alert when job finishes
    experiment.alert(
        title='Main.py Has Finished Running',
        text=f'Runtime: {elapsed_time}')
