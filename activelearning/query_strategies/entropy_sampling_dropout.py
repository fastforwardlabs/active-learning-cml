import numpy as np
import torch
from .strategy import Strategy

"""
Entropy Sampling with Dropout(n_drop times):
Each time -
i) Computes probabilities of class prediction (of all unlabeled training data available) using softmax. Do this n_drop times with dropout turned ON (sampling a different neural network each time) For MNIST, if we used 10k datapoints first round to train a model, we would compute prob using the remaining 50k datapoints.
ii) Take the average prob over n_drop times
iii) Computes entropy of all these points (50k) and select the points with largest entropy
"""


class EntropySamplingDropout(Strategy):
    def __init__(self, X, Y, X_TE, Y_TE, X_NOLB, net, handler, args,
                 n_drop=10):
        """

        """
        super(EntropySamplingDropout, self).__init__(X, Y, X_TE, Y_TE, X_NOLB,
                                                     net, handler, args)

        self.n_drop = n_drop

    def query(self, n):
        # idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        Y_dummy = torch.zeros([self.X_NOLB.shape[0], 1])
        probs = self.predict_prob_dropout(self.X_NOLB,
                                          Y_dummy,
                                          self.n_drop)
        log_probs = torch.log(probs)
        minus_entropy = (probs*log_probs).sum(1)
        data_to_label = self.X_NOLB[minus_entropy.sort()[1][:n]]
        # update self.X_NOLB
        self.X_NOLB = self.X_NOLB[minus_entropy.sort()[1][n:]]
        return data_to_label
