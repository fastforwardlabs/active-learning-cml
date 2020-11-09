import numpy as np
from model import get_net
from dataset import get_dataset, get_handler
from torchvision import transforms
from sklearn.model_selection import train_test_split
import torch
from data import Data
from train import Train
from sample import Sample

if __name__ == "__main__":
    dataset_name = "MNIST"
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.1307,),
                                                              (0.3081,))])
    n_classes = 10
    """
    Get dataset, set data handlers
    """
    X_TR, Y_TR, X_TE, Y_TE = get_dataset(dataset_name)
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
    X_NOLB, X, Y_NOLB, Y = train_test_split(X_TR, Y_TR,
                                            test_size=0.016,
                                            random_state=42,
                                            shuffle=True)
    X_TOLB = torch.empty([10, 28, 28], dtype=torch.uint8)

    data = Data(X, Y, X_TE, Y_TE, X_NOLB, X_TOLB,                                           
            data_transform, handler, n_classes)                                         
  
    data = Data(X, Y, X_TE, Y_TE, X_NOLB,
                data_transform, handler, n_classes)

    # train model
    n_epoch = 5
    train_obj = Train(net, handler, n_epoch, data)
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
