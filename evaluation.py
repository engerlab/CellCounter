import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from dice import dice_loss, dice_coeff


def validation(val_loader, loss_fn, model, device, epoch_num):
    '''
    Validate the model on the validation dataset
    Args:
        val_loader(torch.utils.data.DataLoader): the dataloader of the validation data, initialized
        model (nn.Module): an imported NN model
        device(torch.device): the host device
        epoch_num (int): number of the current epoch the model is on
    Return:
        dice_score (float): the dice score of the prediction
        epoch_val_loss (list): the validation losses for the current epoch
    '''
    epoch_val_loss = []
    model.eval()
    counter=0
    dice_coefficient=0
    for batch in tqdm(val_loader, desc='Validation'): # Obtain one batch of dataset
        img, mask = batch[0].to(device), batch[1].to(device)
        mask = torch.round(mask, decimals=0)
        # mask = one_hot(mask).to(device)

        with torch.no_grad():
            logits = model(img)
            loss = loss_fn(logits.float(), mask.float())
            loss += dice_loss(torch.sigmoid(logits.float()), mask.float(), multiclass=False)
            epoch_val_loss.append(loss.item()) # extract loss value as a Python float

            pred = (torch.sigmoid(logits) > 0.5).float()
            pred = torch.round(pred, decimals=0)
            dice_coefficient += dice_coeff(pred, mask, reduce_batch_first=False)

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
            fig.savefig("./deletepls/prediction" + str(epoch_num) +str(counter) + ".png")
            counter+=1

    dice_score = dice_coefficient / len(val_loader)
    return dice_score, epoch_val_loss



def check_accuracy(loader, model, device=None):
    '''
    Give the accuracy of the model based on pixel to pixel comparison
    Save the pred images
    Args:
        loader(torch.utils.data.DataLoader): the dataloader of the data, initialized
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
