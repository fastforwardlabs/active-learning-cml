import torch
import torch.nn.functional as F
import random


class Sample():
    def __init__(self, clf, data):
        self.clf = clf
        self.data = data

    def predict_prob_unlabeled(self):
        # Y is just a dummy holder for this function so we can use the dataloader
        # X is unlabeled datapool
        loader = self.data.load_nolabel_data()
        self.clf.eval()
        probs = torch.zeros([self.data.X_NOLB.shape[0], self.data.n_classes])
        with torch.no_grad():
            for x, y, idxs in loader:
                out, e1 = self.clf(x)
                # get probabilities by computing softmax of the output
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob
        return probs

    def predict_prob_dropout_unlabeled(self, n_drop):
        # each n_drop is a mask
        # run multiple mask to estimate uncertainty
        loader = self.data.load_nolabel_data()
        # set to train mode to get dropout masks
        self.clf.train()
        # probs = torch.zeros([len(Y), self.n_classes])
        probs = torch.zeros([self.data.X_NOLB.shape[0], self.data.n_classes])
        for i in range(n_drop):
            print('n_drop {}/{}'.format(i+1, n_drop))
            with torch.no_grad():
                for x, y, idxs in loader:
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    # add prob across n_drop
                    probs[idxs] += prob
        probs /= n_drop

        return probs

    def entropy(self, n):
        probs = self.predict_prob_unlabeled()
        log_probs = torch.log(probs)
        # entropy = (-probs*log_probs).sum(1)
        minus_entropy = (probs*log_probs).sum(1)
        data_to_label = self.data.X_NOLB[minus_entropy.sort()[1][:n]]
        X_NOLB = self.data.X_NOLB[minus_entropy.sort()[1][n:]]
        # print(self.X_NOLB.shape[0])
        return data_to_label, X_NOLB

    def entropy_dropout(self, n, n_drop):
        probs = self.predict_prob_dropout_unlabeled(n_drop)
        log_probs = torch.log(probs)
        # entropy = (-probs*log_probs).sum(1)
        minus_entropy = (probs*log_probs).sum(1)
        data_to_label = self.data.X_NOLB[minus_entropy.sort()[1][:n]]
        X_NOLB = self.data.X_NOLB[minus_entropy.sort()[1][n:]]
        # print(self.X_NOLB.shape[0])
        return data_to_label, X_NOLB

    def random(self, n):
        # random shuffle in place
        X_NOLB_tmp = self.data.X_NOLB
        random.shuffle(X_NOLB_tmp)
        # take the top n to label
        data_to_label = X_NOLB_tmp[:n]
        X_NOLB = X_NOLB_tmp[n:]
        return data_to_label, X_NOLB
        
        
