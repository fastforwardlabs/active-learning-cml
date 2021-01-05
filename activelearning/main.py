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

model_dir = "./models/model_" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

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

    #data = Data(X, Y, X_TE, Y_TE, X_NOLB,
    #            data_transform, handler, n_classes)

    # train model
    n_epoch = 10
    train_obj = Train(net, handler, n_epoch, 0.01, data, model_dir)
    train_obj.train()
    # emb = train_obj.get_trained_embedding()
    emb = train_obj.get_test_embedding()
    # select datapoints (need trained model, and data)
    sample = Sample(train_obj.clf, data)
    # get 10 datapoint
    X_TOLB, X_NOLB = sample.entropy(10)
    # update data
    data.update_nolabel(X_NOLB)
    # after labeling
    # data.update_data(X,Y)