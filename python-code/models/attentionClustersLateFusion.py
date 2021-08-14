import torch.nn as nn
import torch
import numpy as np
from torch.nn.parameter import Parameter
import torch.nn.functional as F

"""
    Attention Clusters: Purely Attention Based Local Feature Integration for Video Classification
    Xiang Long, Chuang Gan, Gerard de Melo, Jiajun Wu, Xiao Liu, Shilei Wen.
                                    CVPR 2018
                                    (Late Fusion)
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


class AttentionClustersLateFusion(nn.Module):

    def __init__(self, feature_dim, nclass=2):
        super(AttentionClustersLateFusion, self).__init__()
        # Shifting models for each modality
        self.att_v = ShiftingAttention(feature_dim, 5)
        self.att_f = ShiftingAttention(feature_dim, 5)

        # Regularization for each modality
        self.drop1 = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.1)

        # Fully-connected
        self.fc1 = nn.Linear(feature_dim * 5, nclass)
        self.fc2 = nn.Linear(feature_dim * 5, nclass)

        self.criterion = nn.CrossEntropyLoss().cuda()

    def forward(self, x):
        x_v, weights1 = self.att_v(x[:, :int(x.size(1)/2), :])
        x_f, weights2 = self.att_f(x[:, int(x.size(1)/2):, :])

        x_v = self.drop1(x_v)
        x_f = self.drop2(x_f)

        x_v = self.fc1(x_v)
        x_f = self.fc2(x_f)

        x_v = F.softmax(x_v)
        x_f = F.softmax(x_f)

        negative_scores = torch.zeros_like(x_v)
        positive_scores = torch.zeros_like(x_f)
        fused_scores = torch.zeros_like(x_v)

        # weigthed average
        fused_scores[:, 0] = (8*x_v[:, 0] + 2*x_f[:, 0])/10
        fused_scores[:, 1] = (8*x_v[:, 1] + 2*x_f[:, 1])/10

        return fused_scores, weights1

    def loss(self, outputs, targets):
        loss = self.criterion(outputs, targets)

        return loss
