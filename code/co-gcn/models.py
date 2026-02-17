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
    def __init__(self, num_classes: int = 5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 256, kernel_size=5, padding=2),
            nn.ReLU(),

            nn.Conv1d(256, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.MaxPool1d(8),

            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.classifier = nn.Linear(128 * 22, num_classes)
        
    def forward(self, x, return_features=False):
        x = self.features(x)
        x = torch.flatten(x, 1)
        feats = x
        logits = self.classifier(x)
        if return_features:
            return logits, feats
        return logits



class EEGNet(nn.Module):
    """
    EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces.
    """

    def __init__(self, nb_classes, Chans=64, Samples=128, dropoutRate=0.5,
                 kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25):
        super(EEGNet, self).__init__()

        self.Chans = Chans
        self.Samples = Samples

        # Block 1: Temporal Convolution + Depthwise Spatial Convolution
        self.block1 = nn.Sequential(
            # Padding='same' requires PyTorch >= 1.9
            nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False),
            nn.BatchNorm2d(F1),
            # Depthwise Convolution
            nn.Conv2d(F1, D * F1, (Chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(D * F1),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropoutRate)
        )

        # Block 2: Separable Convolution
        self.block2 = nn.Sequential(
            # Separable Conv Part 1 (Depthwise)
            nn.Conv2d(D * F1, D * F1, (1, 16), padding='same', groups=D * F1, bias=False),
            # Separable Conv Part 2 (Pointwise)
            nn.Conv2d(D * F1, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropoutRate)
        )

        self.flatten = nn.Flatten()

        # Dynamically calculate the size of the linear layer input
        # This prevents errors if you change Chans or Samples
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, Chans, Samples)
            x = self.block1(dummy_input)
            x = self.block2(x)
            x = self.flatten(x)
            n_flatten = x.shape[1]

        self.classifier = nn.Linear(n_flatten, nb_classes)

    def forward(self, x, return_features=False):
        # Input shape: (Batch, Chans, Samples)
        # We need to add the "Channel" dimension for Conv2d: (Batch, 1, Chans, Samples)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.block1(x)
        x = self.block2(x)
        x = self.flatten(x)
        feats = x
        logits = self.classifier(x)
        if return_features:
            return logits, feats
        return logits
    

class EEGNet_tor(nn.Module):
    def __init__(self, nb_classes, Chans=30, Samples=500, dropoutRate=0.5, kernLength=300, F1=8, D=8, F2=64,
                 norm_rate=1.0, dropoutType='Dropout'):
        super(EEGNet_tor, self).__init__()

        # Configure dropout
        self.dropout = nn.Dropout(dropoutRate) if dropoutType == 'Dropout' else nn.Dropout2d(dropoutRate)

        # Block 1
        self.firstConv = nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False)
        self.firstBN = nn.BatchNorm2d(F1)
        self.elu = nn.ELU()

        self.depthwiseConv = nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, padding=0, bias=False)
        self.depthwiseBN = nn.BatchNorm2d(F1 * D)
        self.depthwisePool = nn.AvgPool2d((1, 4))

        # Max-norm should be applied without replacing layer outputs.
        self.depthwiseConv.register_forward_pre_hook(
            self._make_max_norm_pre_hook(norm_rate))

        # Block 2
        self.separableConv = nn.Conv2d(F1 * D, F2, (1, 16), padding='same', bias=False)
        self.separableBN = nn.BatchNorm2d(F2)
        self.separablePool = nn.AvgPool2d((1, 8))

        # Final layers
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(F2 * ((Samples // 4 // 8)), nb_classes)
        self.softmax = nn.Softmax(dim=1)

        self.dense.register_forward_pre_hook(
            self._make_max_norm_pre_hook(norm_rate))

    @staticmethod
    def _make_max_norm_pre_hook(norm_rate):
        def hook(module, _inputs):
            module.weight.data.renorm_(p=2, dim=0, maxnorm=norm_rate)
        return hook

    def forward(self, x, return_features=False):
        x = self.firstConv(x)
        x = self.firstBN(x)
        x = self.elu(x)
        x = self.depthwiseConv(x)
        x = self.depthwiseBN(x)
        x = self.elu(x)
        x = self.depthwisePool(x)
        x = self.dropout(x)
        x = self.separableConv(x)
        x = self.separableBN(x)
        x = self.elu(x)
        x = self.separablePool(x)
        x = self.dropout(x)
        x = self.flatten(x)

        feats = x
        logits = self.softmax(self.dense(x))

        if return_features:
            return logits, feats
        return logits
