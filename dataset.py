import glob
import os
import re
import numpy as np
import monai as mn
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms
from PIL import Image

MASK_SIZE = (452, 452)
class CellDataset(Dataset):
    def __init__(self, img, mask, transform): # initialize object
        self.img = img
        self.mask = mask
        self.transform = transform
        if len(img) != len(mask):
            raise ValueError("Number of images and masks do not match")

    def __len__(self): # len() denotes total number of samples
        return len(self.img)
    
    def __getitem__(self, idx): # get one sample of data at a given index
        sample_name = self.img[idx] # select sample
        mask_name = self.mask[idx] # select corresponding mask

        # Load data and get corresponding mask
        ToTensor = transforms.ToTensor()
        Resize = transforms.Resize(MASK_SIZE)
        Grayscale = transforms.Grayscale()
        X = ToTensor(Image.open(sample_name)) # Already normalized to [0, 1]
        y = ToTensor(Image.open(mask_name))

        if self.transform:
            y = Resize(y)
            y = Grayscale(y)
            X, y = self.transform(X, y)

        return X, y # img and mask


numbers = re.compile(r'(\d+)')
def numericalSort(value): #https://stackoverflow.com/questions/12093940/reading-files-in-a-particular-order-in-python
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def getPaths(dataset, img_or_mask):
    """
    Getter for the paths of different components of the dataset
    Args:
        dataset (str): the name of the cell line in interest
        img_or_mask (str): path is for 'img' or for 'mask'
    Return:
        folder_dir (str): path to the folder containing img or mask
        img_dir (list): list with the name of each sample
    """
    if dataset == 'HCT116':
        folder_dir = os.path.join('..', 'Data', 'HCT116_Dataset', img_or_mask)
        img_dir = sorted(glob.glob(os.path.join(folder_dir, '*.jpg')), key=numericalSort)
        return folder_dir, img_dir


def getLabels(dataset):
    """
    Getter for the labels of different datasets
    Args:
        dataset (str): the name of the cell line in interest
    Return:
        the labels of the dataset in 2D array, number of ground truth colonies in the image in int format 
    """
    if dataset == 'HCT116':
        labels = [113, 90, 99, 87, 96, 93, 92, 95, 96, 76, 60, 63, 10, 11, 10, 1, 1, 1, 
                  145, 175, 160, 167, 156, 120, 395, 446, 436, 107, 101, 92, 43, 28, 45, 4, 3, 6,
                  103, 112, 72, 74, 76, 100, 62, 74, 79, 71, 83, 79, 8, 11, 10, 3, 2, 12,
                  170, 176, 160, 139, 137, 156, 101, 98, 93, 96, 102, 87, 27, 29, 20, 7, 4, 4,
                  165, 159, 129, 149, 164, 160, 90, 89, 103, 116, 127, 128, 18, 18, 16, 3, 8, 9,
                  139, 140, 123, 117, 106, 121, 89, 100, 79, 91, 100, 102, 14, 22, 9, 7, 10, 7,
                  160, 153, 144, 140, 140, 151, 149, 143, 146, 166, 168, 169, 25, 20, 20, 14, 11, 9,
                  157, 160, 117, 138, 140, 155, 144, 118, 117, 138, 125, 108, 27, 29, 25, 6, 11, 10,
                  169, 156, 173, 217, 171, 170, 123, 125, 109, 166, 143, 173, 32, 19, 21, 8, 6, 4,
                  153, 156, 135, 157, 138, 163, 94, 92, 72, 119, 117, 92, 31, 31, 21, 4, 9, 8,
                  147, 158, 143, 156, 149, 119, 118, 143, 121, 168, 175, 144, 28, 27, 18, 10, 8, 13,
                  137, 115, 115, 126, 122, 136, 93, 105, 114, 137, 119, 142, 28, 29, 25, 8, 12, 7,
                  228, 240, 294, 259, 243, 239, 1, 1, 0,
                  114, 98, 115, 104, 98, 121, 99, 75, 75, 59, 76, 62, 9, 9, 14, 4, 1, 0,
                  220, 202, 213, 176, 209, 201, 112, 127, 101, 65, 62, 64, 18, 23, 31, 5, 3, 5,
                  145, 152, 167, 164, 167, 166, 114, 82, 74, 59, 12, 10, 13, 1, 3, 4,
                  61, 59, 62, 60, 67, 57, 30, 19, 36, 33, 45, 32, 6, 7, 16, 16, 16, 7, 15, 15, 17, 33, 28, 28,
                  103, 95, 87, 87, 83, 90, 44, 48, 51, 45, 63, 55, 13, 19, 17, 19, 20, 21, 6, 7, 2, 4, 2, 4,
                  74, 72, 93, 65, 76, 87, 60, 55, 49, 59, 49, 66, 16, 11, 9, 15, 12, 14, 4, 2, 5, 7, 4, 8,
                  8, 11, 13, 6, 10, 5, 46, 55, 53, 69, 67, 52, 8, 11, 14, 6, 19, 12, 12, 18, 11, 14, 15, 11,
                  70, 90, 80, 96, 93, 76, 68, 63, 65, 78, 75, 59, 10, 6, 10, 8, 16, 9, 3, 2, 4, 6, 5, 4,
                  43, 48, 46, 34, 41, 41, 12, 16, 8, 25, 24, 20, 3, 7, 12, 6, 7, 5, 4, 2, 2, 1, 2, 0,
                  133, 130, 99, 105, 117, 110, 34, 49, 55, 48, 55, 65, 17, 22, 14, 14, 16, 13, 2, 2, 3, 8, 1, 6,
                  29, 40, 5, 12, 9, 6,
                  118, 102, 108, 101, 112, 113, 46, 45, 35, 47, 54, 45, 11, 16, 16, 12, 20, 17, 3, 2, 1, 4, 3, 1,
                  144, 117, 147, 149, 125, 122, 56, 73, 52, 40, 55, 51, 21, 29, 30, 23, 19, 32, 9, 5, 3, 8, 3, 5,
                  131, 148, 160, 155, 158, 132, 96, 110, 79, 116, 123, 108, 46, 33, 46, 40, 45, 42, 42, 26, 40, 42, 30, 36, #27
                  157, 144, 143, 142, 159, 163, 78, 78, 70, 149, 153, 124, 53, 43, 44, 49, 45, 45, 28, 26, 22, 24, 20, 20,
                  134, 128, 108, 121, 111, 105, 94, 85, 84, 110, 126, 116, 53, 48, 49, 51, 51, 40, 41, 46, 36, 38, 42, 37,
                  159, 134, 163, 165, 179, 147, 94, 95, 92, 158, 149, 158, 41, 39, 39, 38, 45, 47, 36, 30, 32, 34, 36, 25,
                  181, 136, 161, 158, 155, 130, 70, 74, 72, 128, 140, 125, 51, 42, 46, 52, 57, 48, 18, 25, 27, 20, 23, 19,
                  137, 141, 137, 147, 158, 124, 82, 75, 80, 166, 143, 130, 28, 34, 39, 47, 39, 46, 17, 25, 21, 19, 22, 20,
                  146, 146, 164, 125, 150, 160, 111, 110, 98, 134, 120, 129, 64, 46, 52, 60, 43, 55, 36, 34, 42, 31, 28, 36,
                  109, 107, 110, 78, 139, 120, 40, 57, 73, 60, 65, 66, 14, 29, 31, 26, 22, 25, 8, 8, 10, 11, 16, 7,
                  131, 114, 139, 110, 125, 123, 104, 105, 118, 83, 95, 90, 13, 19, 18, 15, 17, 12, 10, 12, 7, 7, 7, 7,
                  114, 164, 154, 141, 129, 118, 112, 122, 119, 160, 184, 175, 30, 36, 34, 17, 33, 33, 19, 17, 8, 8, 17, 18,
                  119, 130, 132, 135, 124, 140, 87, 107, 109, 132, 134, 100, 22, 14, 22, 19, 23, 17, 10, 4, 8, 9, 6, 4,
                  146, 143, 127, 148, 123, 112, 81, 102, 82, 90, 106, 101, 35, 35, 28, 31, 34, 28, 9, 9, 8, 9, 11, 9]
        return labels
