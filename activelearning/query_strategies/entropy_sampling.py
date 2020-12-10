import numpy as np
import torch
from .strategy import Strategy


"""
Entropy Sampling:
i) Computes probabilities of class prediction (of all unlabeled training data available) using softmax.
For MNIST, if we used 10k datapoints first round to train a model, we would compute prob using the remaining 50k datapoints
ii) Computes entropy of all these points (50k) and select the points with largest entropy
"""


class EntropySampling(Strategy):
    
    def __init__(self, X, Y, X_TE, Y_TE, X_NOLB, net, handler, args):
        super(EntropySampling, self).__init__(X, Y, X_TE, Y_TE, X_NOLB, net,
                                              handler, args)

    def query(self, n):
        Y_dummy = torch.zeros([self.X_NOLB.shape[0], 1])
        probs = self.predict_prob(self.X_NOLB, Y_dummy)
        log_probs = torch.log(probs)
        # entropy = (-probs*log_probs).sum(1)
        minus_entropy = (probs*log_probs).sum(1)      
        """
        sort small to large, take smallest n 
        vs largest because of the inversion in computing entropy
        """
        # print(minus_entropy.sort()[1][:n])
        data_to_label = self.X_NOLB[minus_entropy.sort()[1][:n]]
        # update self.X_NOLB
        self.X_NOLB = self.X_NOLB[minus_entropy.sort()[1][n:]]
        # print(self.X_NOLB.shape[0])
        return data_to_label
