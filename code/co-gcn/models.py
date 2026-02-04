import math
import torch
import numpy as np
from scipy.special import softmax
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class MixedGraphConvolution(nn.Module):
    """
    Mixed CoGCN layer
    """
    def __init__(self, in_features, out_features, adjs, n_view, lr_alpha, device, bias=True):
        super(MixedGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.alpha = np.ones(n_view) / n_view
        self.adjs = adjs
        self.n = self.adjs[0].size()[0]
        self.adj = torch.sparse.FloatTensor(self.n, self.n)
        self.H = None
        self.input = None
        self.lr_alpha = lr_alpha
        self.device = device

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
            nn.init.uniform_(self.bias)
        else:
            self.register_parameter('bias', None)
        self._reset_parameters()
        self._update_adj()
        self.n_view = n_view

    def _reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def _update_adj(self):
        np.clip(self.alpha, 0.0, 1.0)
        self.alpha = softmax(self.alpha)
        self.adj = torch.sparse.FloatTensor(self.n, self.n).to(self.device)
        for alpha, adj in zip(self.alpha, self.adjs):
            self.adj.add_(alpha * adj)

    def update_alpha(self):
        for i in range(self.n_view):
            support = torch.spmm(self.adjs[i], self.input)
            partial_H = torch.mm(support, self.weight)
            gHT = torch.transpose(self.H.grad, 0, 1)
            self.alpha[i] -= self.lr_alpha * torch.trace(torch.mm(gHT, partial_H)).item()
            self._update_adj()
            step = self.lr_alpha * torch.trace(torch.mm(gHT, partial_H)).item()
            # print(i, "step", step, "alpha_before", self.alpha[i])

    def forward(self, input):
        self.input = input
        support = torch.mm(input, self.weight)
        self.H = torch.sparse.mm(self.adj, support)
        if self.H.requires_grad:
            self.H.retain_grad()
        if self.bias is not None:
            return self.H + self.bias
        else:
            return self.H

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class CoGCN(nn.Module):
    def __init__(self, nfeat, nclass, adjs, n_view, lr_alpha, dropout, device):
        super(CoGCN, self).__init__()
        nhid = max(nfeat // 1280 * 64, 64)
        self.gc1 = MixedGraphConvolution(nfeat, nhid, adjs, n_view, lr_alpha, device)
        self.gc2 = MixedGraphConvolution(nhid, nclass, adjs, n_view, lr_alpha, device)
        self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.gc1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x)
        return F.log_softmax(x, dim=1)

class AudioSCNN(nn.Module):
    def __init__(self, n_classes=5):
        super(AudioSCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=256, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=5, padding=2)

        self.dropout1 = nn.Dropout(0.1)

        self.maxpool1 = nn.MaxPool1d(kernel_size=8, stride=8)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, padding=2)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, padding=2)

        self.dropout2 = nn.Dropout(0.5)

        self.classifier = nn.Linear(in_features=128 * 22, out_features=n_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x, return_features=False):
        x = self.relu(self.conv1(x)) # [, 1, 180] - > [, 256, 180]
        x = self.relu(self.conv2(x)) # [, 256, 180] - > [, 128, 180]
        x = self.dropout1(x) # [, 128, 180] -> [, 128, 180]
        x = self.maxpool1(x) # [, 128, 180] -> [, 128, 22]
        x = self.relu(self.conv3(x)) # [, 128, 22] - > [, 128, 22]
        x = self.relu(self.conv4(x)) # [, 128, 22] - > [, 128, 22]
        x = self.dropout2(x) # [, 128, 22] -> [, 128, 22]
        x = torch.flatten(x, 1) # [, 128, 22] -> [, 128x22]
        feats = x
        logits = self.classifier(x) # [, 128x22] -> [, 5]
        if return_features:
            return logits, feats
        return logits
    
class EEGNet(nn.Module):
    def __init__(self, nb_classes, Chans=64, Samples=128, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16,
                 norm_rate=0.25, dropoutType='Dropout'):
        super(EEGNet, self).__init__()

        if dropoutType not in ('Dropout', 'SpatialDropout2D'):
            raise ValueError('dropoutType must be one of SpatialDropout2D or Dropout, passed as a string.')
        self.dropout = nn.Dropout(dropoutRate) if dropoutType == 'Dropout' else nn.Dropout2d(dropoutRate)

        # Block 1
        self.firstConv = nn.Conv2d(1, F1, (1, kernLength), padding="same", bias=False)
        self.firstBN = nn.BatchNorm2d(F1)

        self.depthwiseConv = nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, padding=0, bias=False)
        self.depthwiseBN = nn.BatchNorm2d(F1 * D)
        self.depthwisePool = nn.AvgPool2d((1, 4))

        # Block 2: SeparableConv2D = depthwise (1x16) + pointwise (1x1)
        self.separableDepth = nn.Conv2d(F1 * D, F1 * D, (1, 16), groups=F1 * D, padding="same", bias=False)
        self.separablePoint = nn.Conv2d(F1 * D, F2, (1, 1), padding=0, bias=False)
        self.separableBN = nn.BatchNorm2d(F2)
        self.separablePool = nn.AvgPool2d((1, 8))

        self.elu = nn.ELU()

        # Final layers
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(F2 * (Samples // 4 // 8), nb_classes)

        # Max-norm constraints
        self.depthwise_maxnorm = 1.0
        self.dense_maxnorm = norm_rate

    def _apply_maxnorm(self):
        # Match Keras max_norm constraint behavior on weights
        if self.depthwise_maxnorm is not None:
            self.depthwiseConv.weight.data.renorm_(p=2, dim=0, maxnorm=self.depthwise_maxnorm)
        if self.dense_maxnorm is not None:
            self.dense.weight.data.renorm_(p=2, dim=0, maxnorm=self.dense_maxnorm)

    def forward(self, x, return_features=False):
        self._apply_maxnorm()

        x = self.firstConv(x)
        x = self.firstBN(x)
        x = self.depthwiseConv(x)
        x = self.depthwiseBN(x)
        x = self.elu(x)
        x = self.depthwisePool(x)
        x = self.dropout(x)

        x = self.separableDepth(x)
        x = self.separablePoint(x)
        x = self.separableBN(x)
        x = self.elu(x)
        x = self.separablePool(x)
        x = self.dropout(x)

        x = self.flatten(x)
        feats = x
        logits = self.dense(x)
        if return_features:
            return logits, feats
        return logits
