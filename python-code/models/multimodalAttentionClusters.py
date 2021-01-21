import torch.nn as nn
import torch
import numpy as np
from torch.nn.parameter import Parameter
import torch.nn.functional as F

"""
    Attention Clusters: Purely Attention Based Local Feature Integration for Video Classification
    Xiang Long, Chuang Gan, Gerard de Melo, Jiajun Wu, Xiao Liu, Shilei Wen.
                                    CVPR 2018
"""


def get_fans(shape):
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        fan_in = np.sqrt(np.prod(shape))
        fan_out = np.sqrt(np.prod(shape))
    return fan_in, fan_out


def normal(shape, scale=0.05):
    tensor = torch.FloatTensor(*shape)
    tensor.normal_(mean=0.0,  std=scale)
    return tensor


def glorot_normal(shape):
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(2. / (fan_in + fan_out))
    return normal(shape, s)

_softmax = nn.Softmax()


def softmax_m1(x):
    flat_x = x.view(-1, x.size(-1))
    flat_y = _softmax(flat_x)
    y = flat_y.view(*x.size())
    return y


class ShiftingAttention(nn.Module):

    def __init__(self, dim, n):
        super(ShiftingAttention, self).__init__()
        self.dim = dim
        self.n_att = n

        self.attentions = nn.Conv1d(dim, n, 1, bias=True)
        self.gnorm = np.sqrt(n)

        self.w = Parameter(glorot_normal((n,)))
        self.b = Parameter(glorot_normal((n,)))

    def forward(self, x):
        '''x = (N, L, F)'''
        scores = self.attentions(torch.transpose(x, 1, 2))
        '''scores = (N, C, L)'''
        weights = softmax_m1(scores)
        '''weights = (N, C, L), sum(weights, -1) = 1'''

        outs = []
        for i in range(self.n_att):
            weight = weights[:, i, :]
            ''' weight = (N, L) '''
            weight = weight.unsqueeze(-1).expand_as(x)
            ''' weight = (N, L, F) '''

            w = self.w[i].unsqueeze(0).expand(x.size(0), x.size(-1))
            b = self.b[i].unsqueeze(0).expand(x.size(0), x.size(-1))
            ''' center = (N, L, F) '''

            o = torch.sum(x * weight, 1).squeeze(1) * w + b

            norm2 = torch.norm(o, 2, -1, keepdim=True).expand_as(o)
            o = o / norm2 / self.gnorm
            outs.append(o)
        outputs = torch.cat(outs, -1)
        '''outputs = (N, F*C)'''
        return outputs, weights


class AttentionClustersMultimodal(nn.Module):

    def __init__(self, feature_dim, nclass=2):
        super(AttentionClustersMultimodal, self).__init__()
        # Shifting models for each modality
        self.att_v = ShiftingAttention(feature_dim, 5)
        self.att_f = ShiftingAttention(feature_dim, 5)

        # Regularization for each modality
        self.dropout = nn.Dropout(0.1)

        # Fully-connected
        self.fc = nn.Linear(feature_dim * 5 * 2, nclass)

        self.criterion = nn.CrossEntropyLoss().cuda()

    def forward(self, x):
        x_v, weights1 = self.att_v(x[:, :150, :])
        x_f, weights2 = self.att_f(x[:, 150:, :])

        x = torch.cat((x_v, x_f), 1)
        x = self.dropout(x)

        x = self.fc(x)
        return F.softmax(x), weights1

    def loss(self, outputs, targets):
        loss = self.criterion(outputs[0], targets)

        return loss
