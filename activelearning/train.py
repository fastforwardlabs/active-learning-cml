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
import os
import csv
import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from collections import deque 

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
        self.step = deque() 
        self.train_loss = deque() 
        self.val_loss = deque() 
        self.train_acc = deque() 
        self.val_acc = deque() 
        self.clf = self.clf().to(self.device)
        
    def save_checkpoint(self, checkpoint, model_dir):
        f_path = os.path.join(model_dir, 'checkpoint.pt')
        
        torch.save(checkpoint, f_path)

    def train(self):
        print('train:train with {} datapoints'.format(self.data.X.shape[0]))
        checkpoint_fpath = os.path.join(self.model_dir, 'checkpoint.pt')
        #print(checkpoint_fpath)
        #self.clf = self.clf(n_classes=self.data.n_classes).to(self.device)
        
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
        
        train_loader = self.data.load_train_data()
        test_loader = self.data.load_test_data()
        
        runlog_filename = os.path.join(self.model_dir, "run_log.csv")
        fieldnames = ['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc']
        if not os.path.isfile(runlog_filename):
            csvfile = open(runlog_filename, 'w', newline='', )
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        else: # else it exists so append without writing the header
            csvfile = open(runlog_filename, 'a', newline='')
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
        for epoch in range(start_epoch, start_epoch + self.n_epoch):
            self._train(train_loader, optimizer, epoch)
            self._test(test_loader, epoch)
            
            writer.writerow({'epoch': self.step[-1], 'train_loss': self.train_loss[-1],
                            'val_loss': self.val_loss[-1], 'train_acc': self.train_acc[-1],
                            'val_acc': self.val_acc[-1]})            
        checkpoint = {
            'epoch': self.n_epoch + start_epoch,
            'state_dict': self.clf.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        self.save_checkpoint(checkpoint, self.model_dir)

    def _train(self, train_loader, optimizer, epoch):
        self.clf.train()
        train_loss = 0
        correct = 0
        
        for batch_idx, (data, target, idxs) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output, _ = self.clf(data)
            loss = F.nll_loss(output, target)
            train_loss += F.nll_loss(output, target,  reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            loss.backward()
            optimizer.step()
            
        train_loss /= len(train_loader.dataset)

        print('\nTrain set: Epoch: {}, Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            epoch, train_loss, correct, len(train_loader.dataset),
            100. * correct / len(train_loader.dataset)))
        
        self.step.append(epoch)
        self.train_loss.append(train_loss)
        self.train_acc.append(correct / len(train_loader.dataset))

    def _test(self, test_loader, epoch):
        self.clf.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target, idxs) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output, _ = self.clf(data)
                loss = F.nll_loss(output, target)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)

        print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            epoch, test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        self.val_loss.append(test_loss)
        self.val_acc.append(correct / len(test_loader.dataset))

    
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