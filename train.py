import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


class Train:
    def __init__(self, net, handler, n_epoch, data):
        self.data = data
        self.clf = net
        self.handler = handler
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.n_epoch = n_epoch

    def train(self):
        print('train:train with {} datapoints'.format(self.data.X.shape[0]))
        self.clf = self.clf(n_classes=self.data.n_classes).to(self.device)
        optimizer = optim.SGD(self.clf.parameters(), lr=0.01, momentum=0.5)
        loader_tr = self.data.load_train_data()
        for epoch in range(1, self.n_epoch+1):
            train_loss = self._train(loader_tr, optimizer)
            #_, test_loss = self.predict(self.data.X_TE, self.data.Y_TE)
            _, test_loss = self.predict_test()
            train_acc = self.check_accuracy(split='train')
            test_acc = self.check_accuracy(split='test')
            print("{}\t{}\t\t{}\t\t{}\t\t{}".format(epoch,
                                                    round(train_loss, 4),
                                                    round(test_loss, 4),
                                                    round(train_acc, 6),
                                                    round(test_acc, 6)))

    def _train(self, loader_tr, optimizer):
        self.clf.train()
        total_loss = 0
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            '''
            print('_train x shape {}'.format(x.shape))
            print('_train y shape {}'.format(y.shape))
            '''
            # x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            out, e1 = self.clf(x)
            loss = F.cross_entropy(out, y)
            total_loss += loss.cpu().item()
            loss.backward()
            optimizer.step()
            
        return total_loss/len(loader_tr)

    def check_accuracy(self, split='test'):
        if split == 'test':
            loader = self.data.load_test_data()
        elif split == 'train':
            loader = self.data.load_train_data()
        self.clf.eval()
        num_correct, num_samples = 0, 0
        
        for x, y, idxs in loader:
            # x, y = x.to(self.device), y.to(self.device)
            scores, e1 = self.clf(x)
            _, preds = scores.data.cpu().max(1)
            #if str(self.device) == 'cuda':
            #   y = y.cpu()
            num_correct += (preds == y).sum()
            num_samples += x.size(0)
        
        # Return the fraction of datapoints that were correctly classified.
        acc = float(num_correct) / num_samples
        return acc    

    '''
    def load_train_data(self):
        loader = DataLoader(self.handler(self.data.X,
                                         self.data.Y,
                                         transform=self.data.transform),
                            shuffle=True,
                            batch_size=64,
                            num_workers=1)
        return loader

    def load_test_data(self):
        loader = DataLoader(self.handler(self.data.X,
                                         self.data.Y,
                                         transform=self.data.transform),
                            shuffle=False,
                            batch_size=1000,
                            num_workers=1)
        return loader
    '''
    def predict_test(self):
        loader_te = self.data.load_test_data()
        self.clf.eval()
        total_loss = 0
        P = torch.zeros(len(self.data.Y_TE), dtype=self.data.Y_TE.dtype)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                """
                print('prediction x shape {}'.format(x.shape))
                print('prediction y shape {}'.format(y.shape))
                """
                # x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                loss = F.cross_entropy(out, y)
                total_loss += loss.cpu().item()
                pred = out.max(1)[1]
                # if str(self.device) == 'cuda':
                #    P[idxs] = pred.cpu()
                #else:
                P[idxs] = pred
    
        return P, total_loss/len(loader_te)

    def get_trained_embedding(self):
        loader = self.data.load_train_data()
        self.clf.eval()
        emb = np.zeros((self.data.X.shape[0], self.clf.get_embedding_dim()))
        with torch.no_grad():
            for x, y, idxs in loader:
                # x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                emb[idxs] = e1

        return emb

    def get_prev_trained_embedding(self):
        loader = self.data.load_prev_data()
        self.clf.eval()
        emb = np.zeros((self.data.X_Prev.shape[0],
                        self.clf.get_embedding_dim()))
        with torch.no_grad():
            for x, y, idxs in loader:
                # x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                emb[idxs] = e1

        return emb

    def get_test_embedding(self):
        loader = self.data.load_test_data()
        self.clf.eval()
        emb = np.zeros((self.data.X_TE.shape[0], self.clf.get_embedding_dim()))
        with torch.no_grad():
            for x, y, idxs in loader:
                out, e1 = self.clf(x)
                emb[idxs] = e1

        return emb

    def get_nolb_embedding(self):
        loader = self.data.load_nolabel_data()
        self.clf.eval()
        emb = np.zeros((self.data.X_NOLB.shape[0], self.clf.get_embedding_dim()))
        with torch.no_grad():
            for x, y, idxs in loader:
                out, e1 = self.clf(x)
                emb[idxs] = e1

        return emb

    def get_tolb_embedding(self):
        loader = self.data.load_tolabel_data()
        self.clf.eval()
        emb = np.zeros((self.data.X_TOLB.shape[0], self.clf.get_embedding_dim()))
        with torch.no_grad():
            for x, y, idxs in loader:
                out, e1 = self.clf(x)
                emb[idxs] = e1
        return emb

    def update_data(self, data):
        self.data = data
        
