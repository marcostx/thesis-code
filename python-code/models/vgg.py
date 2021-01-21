import torch.nn as nn
import torch
import numpy as np
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torchvision import models


class VGG19(nn.Module):

    def __init__(self, hiddden_dim, nclass=2):
        super(VGG19, self).__init__()
        self.descriptor = models.vgg19(pretrained=True)
        # for param in self.descriptor.parameters():
        #     param.requires_grad=False
        #
        self.descriptor.classifier[6] = nn.Linear(hiddden_dim, nclass)

        self.criterion = nn.CrossEntropyLoss().cuda()

    def forward(self, frame):
        outs = self.descriptor(frame)

        return outs

    def loss(self, outputs, targets):
        loss = self.criterion(outputs, targets)

        return loss
