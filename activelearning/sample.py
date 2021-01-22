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
        
        
