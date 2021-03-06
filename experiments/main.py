# ###########################################################################
#
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2021
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

"""
Code to experiment and test the model training w/o UI (Dash part of it)
"""
import numpy as np
import torch
import umap
import os
import datetime

from torchvision import transforms
from sklearn.model_selection import train_test_split
from collections import deque 

from activelearning.model import get_net
from activelearning.dataset import get_dataset, get_handler
from activelearning.data import Data
from activelearning.train import Train
from activelearning.sample import Sample

seed = 123

"""
Set random seeds for UMAP
Initialize UMAP reducer
"""
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if not os.path.exists("models"):
    os.mkdir("models")
model_dir = "models/model_" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

if __name__ == "__main__":
    dataset_name = "MNIST"
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.1307,),
                                                              (0.3081,))])
    n_classes = 10
    """
    Get dataset, set data handlers
    """
    _, _, X_INIT, Y_INIT = get_dataset(dataset_name)

    handler = get_handler(dataset_name)

    """
    Define model architecture
    """
    net = get_net(dataset_name)

    """
    Split train into labeled and unlabeled
    X_NOLB is unlabeled pool, Y_NOLB is the true label (unused)
    X is the labeled pool we can use to train a base model
    Y is the target
    Initialize X_TOLB (these are the datapoints to be labeled by human
    """
    X_TR, X_TE, Y_TR, Y_TE = train_test_split(X_INIT, Y_INIT,
                                              test_size=0.1024,
                                              random_state=seed,
                                              shuffle=True)
    train_ratio = 1024/X_TR.shape[0]
    print("train_ratio", train_ratio)
    X_NOLB, X, Y_NOLB, Y = train_test_split(X_TR, Y_TR,
                        test_size=train_ratio,
                        random_state=seed,
                        shuffle=True)
    X_TOLB = torch.empty([10, 28, 28], dtype=torch.uint8)
    '''
    Make data and train objects
    '''
    data = Data(X, Y, X_TE, Y_TE, X_NOLB, X_TOLB, 
                data_transform, 
                handler, n_classes)     
    print("data.X: ", data.X.shape)
    print("data.X_TE: ", data.X_TE.shape)

    # train model
    n_epoch = 10
    train_obj = Train(net, handler, n_epoch, 0.01, data, model_dir)
    train_obj.train()
    emb = train_obj.get_trained_embedding()
    #emb = train_obj.get_test_embedding()
    # select datapoints (need trained model, and data)
    sample = Sample(train_obj.clf, data)
    # get 10 datapoint
    X_TOLB, X_NOLB = sample.entropy(10)
    # update data
    data.update_nolabel(X_NOLB)
    # after labeling
    # data.update_data(X,Y)