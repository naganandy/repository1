import torch.nn as nn
import torch, numpy as np
import torch.nn.functional as F
from model import layers 
from torch.autograd import Variable

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, args):
        super(GCN, self).__init__()

        self.gc1 = layers.GraphConvolution(nfeat, nhid)
        self.gc2 = layers.GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x1 = x
        x = self.gc2(x, adj)
        return F.log_softmax(x), x1


class MLP(nn.Module):
    def __init__(self, inputLen, nhid, nclass, dropout, Cuda):
        super(MLP, self).__init__()
        
        self.inputLen = inputLen
        self.nclass = nclass
        self.Cuda = Cuda
        self.dropout = dropout
        self.fc1 = torch.nn.Linear(2 * inputLen, nhid)
        self.fc2 = torch.nn.Linear(nhid, nclass)

    def forward(self, embPairs, test=False):
        catEmb = np.zeros((len(embPairs), 2 * self.inputLen))

        for i, (e1, e2) in enumerate(embPairs):
            concat = np.concatenate((e1, e2), 0)
            catEmb[i] = concat

        catEmb = Variable(torch.FloatTensor(catEmb))

        if self.Cuda:
            catEmb = catEmb.cuda()

        hidden = F.relu(self.fc1(catEmb))
        hidden = F.dropout(hidden, self.dropout, training = self.training)
        out = F.log_softmax(self.fc2(hidden))
        return out