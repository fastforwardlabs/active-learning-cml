import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms


class Data:
    def __init__(self, X, Y, X_TE, Y_TE, X_NOLB, X_TOLB, data_transform, handler, n_classes):
        self.X = X
        self.Y = Y
        self.X_NOLB = X_NOLB
        self.X_TOLB = X_TOLB
        self.X_TE = X_TE
        self.Y_TE = Y_TE
        self.X_prev = X  # initialize
        self.Y_prev = Y  # initialize
        self.transform = data_transform
        self.handler = handler
        self.n_classes = n_classes
        
    def update_data(self, X, Y):
        self.X_prev = self.X
        self.Y_prev = self.Y
        self.X = X
        self.Y = Y
    
    def update_nolabel(self, X_NOLB):
        self.X_NOLB = X_NOLB

    def update_tolabel(self, X_TOLB):
        self.X_TOLB = X_TOLB

    def load_train_data(self):
        loader = DataLoader(self.handler(self.X,
                                         self.Y,
                                         transform=self.transform),
                            shuffle=True,
                            batch_size=64,
                            num_workers=1)
        return loader

    def load_test_data(self):
        loader = DataLoader(self.handler(self.X_TE,
                                         self.Y_TE,
                                         transform=self.transform),
                            shuffle=False,
                            batch_size=64,
                            num_workers=1)
        return loader
    
    def load_nolabel_data(self):
        Y_dummy = torch.zeros([self.X_NOLB.shape[0], 1])
        loader = DataLoader(self.handler(self.X_NOLB,
                                         Y_dummy,
                                         transform=self.transform),
                            shuffle=False,
                            batch_size=1000,
                            num_workers=1)
        return loader

    def load_tolabel_data(self):
        Y_dummy = torch.zeros([self.X_TOLB.shape[0], 1])
        loader = DataLoader(self.handler(self.X_TOLB,
                                         Y_dummy,
                                         transform=self.transform),
                            shuffle=False,
                            batch_size=1000,
                            num_workers=1)
        return loader

    def load_prev_data(self):
        loader = DataLoader(self.handler(self.X_Prev,
                                         self.Y_Prev,
                                         transform=self.transform),
                            shuffle=True,
                            batch_size=1000,
                            num_workers=1)
        return loader
