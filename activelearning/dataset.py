import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import os.path
import errno


def get_dataset(name):
    if name == 'MNIST':
        return get_MNIST()

def get_MNIST():
    raw_tr = datasets.MNIST('./MNIST', train=True, download=True)
    raw_te = datasets.MNIST('./MNIST', train=False, download=True)
    X_tr = raw_tr.data
    Y_tr = raw_tr.targets
    X_te = raw_te.data
    Y_te = raw_te.targets
    return X_tr, Y_tr, X_te, Y_te

def get_handler(name):
    if name == 'MNIST':
        return MNISTHandler

class MNISTHandler(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x.numpy(), mode='L')
            x = self.transform(x)
            #print("x.shape: ", x.shape)
        return x, y, index

    def __len__(self):
        return len(self.X)
