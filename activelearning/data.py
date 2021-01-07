# ###########################################################################
#
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2020
#  All rights reserved.
#
#  Applicable Open Source License: Apache 2.0
#
#  NOTE: Cloudera open source products are modular software products 
#  made up of hundreds of individual components, each of which was 
#  individually copyrighted.  Each Cloudera open source product is a 
#  collective work under U.S. Copyright Law. Your license to use the 
#  collective work is as provided in your written agreement with  
#  Cloudera.  Used apart from the collective work, this file is 
#  licensed for your use pursuant to the open source license 
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute 
#  this code. If you do not have a written agreement with Cloudera nor 
#  with an authorized and properly licensed third party, you do not 
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED 
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO 
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND 
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU, 
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS 
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE 
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES 
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF 
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# ###########################################################################

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
