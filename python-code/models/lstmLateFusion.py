import torch
from torch import nn
import torch.nn.functional as F
from .cbam import *
from models.base_model import BaseModel
from torchvision import models
from collections import OrderedDict
from models.lstm import LSTM

class LSTMLateFusion(BaseModel):

    def __init__(self, model_name, input_dim, att_type, hidden_dim, if_attention, dropout_prob=0.5, output_dim=2,
                 num_layers=1):
        super(LSTMLateFusion, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.if_att = if_attention
        self.att_type = att_type
        self.dropout_prob = dropout_prob

        self.lstm_rgb = LSTM(model_name, input_dim, att_type, hidden_dim, if_attention)
        self.lstm_flow = LSTM(model_name, input_dim, att_type, hidden_dim, if_attention)

        self.criterion = nn.CrossEntropyLoss().cuda()

    def forward(self, x):
        x_v, weights = self.lstm_rgb(x[:, :int(x.size(1)/2), :])
        x_f, weights = self.lstm_flow(x[:, int(x.size(1)/2):, :])

        negative_scores = torch.zeros_like(x_v)
        positive_scores = torch.zeros_like(x_f)
        fused_scores = torch.zeros_like(x_v)
        
        # negative_scores[:,0] = x_v[:,0]
        # negative_scores[:,1] = x_f[:,0]

        # positive_scores[:,0] = x_v[:,1]
        # positive_scores[:,1] = x_f[:,1]

        # # 0.5 importance for each modality        
        # fused_scores[:,0] = torch.mean(negative_scores,dim=1)
        # fused_scores[:,1] = torch.mean(positive_scores,dim=1)
        fused_scores[:,0] = (8*x_v[:,0] + 2*x_f[:,0])/10
        fused_scores[:,1] = (8*x_v[:,1] + 2*x_f[:,1])/10
        
        return fused_scores, weights    

    def loss(self, outputs, targets):
        loss = self.criterion(outputs, targets)

        return loss
