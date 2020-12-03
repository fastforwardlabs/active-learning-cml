import numpy as np
import os
import csv
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


class Train:
    def __init__(self, net, handler, n_epoch, lr, data, model_dir):
        self.data = data
        self.clf = net
        self.handler = handler
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.n_epoch = n_epoch
        self.lr = lr
        self.model_dir = model_dir
        
    def save_checkpoint(self, checkpoint, model_dir):
        f_path = os.path.join(model_dir, 'checkpoint.pt')
        
        torch.save(checkpoint, f_path)

    def train(self):
        print('train:train with {} datapoints'.format(self.data.X.shape[0]))
        checkpoint_fpath = os.path.join(self.model_dir, 'checkpoint.pt')
        #print(checkpoint_fpath)
        self.clf = self.clf(n_classes=self.data.n_classes).to(self.device)
        optimizer = optim.SGD(self.clf.parameters(), lr=self.lr, momentum=0.5)
        start_epoch = 1
        
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
            
        # load checkpoint if available
        if os.path.isfile(checkpoint_fpath):
            checkpoint = torch.load(checkpoint_fpath)
            self.clf.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
        
        loader_tr = self.data.load_train_data()
        
        runlog_filename = os.path.join(self.model_dir, "run_log.csv")
        fieldnames = ['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc']
        if not os.path.isfile(runlog_filename):
            csvfile = open(runlog_filename, 'w', newline='', )
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        else: # else it exists so append without writing the header
            csvfile = open(runlog_filename, 'a', newline='')
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
        for epoch in range(start_epoch, start_epoch+self.n_epoch):
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
            writer.writerow({'epoch': epoch, 'train_loss': train_loss,
                            'val_loss': test_loss, 'train_acc': train_acc,
                            'val_acc': test_acc})
            
        checkpoint = {
            'epoch': self.n_epoch + start_epoch,
            'state_dict': self.clf.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        self.save_checkpoint(checkpoint, self.model_dir)
        

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