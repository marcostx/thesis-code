from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from .cbam import *
from models.base_model import BaseModel
from torch.nn import init


def weight_model(att_rnn_size, att_sig_w):
    model = nn.Sequential(
        nn.Linear(2*att_rnn_size, 1),
        nn.MulConstant(att_sig_w),
        nn.Sigmoid()
    )
    return model

class TAGM_model(BaseModel):
    def __init__(self, input_sz, hidden_sz):
        super(TAGM_model,self).__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.i2h = nn.Linear(input_sz, hidden_sz)
        self.h2h = nn.Linear(hidden_sz, hidden_sz)

    def forward(self, x, att_weights, prev_h):

        forget = 1-att_weights
        i2h = self.i2h(x)
        h2h = self.h2h(prev_h)
        in_transf = F.relu(i2h+h2h)

        next_h = forget*prev_h + att_weights*in_transf
        hidden_v = F.dropout(next_h, p=0.25)

        return hidden_v


# Here we define our model as a class
class Attention(BaseModel):

    def __init__(self, input_dim, att_rnn_size, dropout=0.5,
                    num_layers=1):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.att_rnn_size = att_rnn_size
        self.num_layers = num_layers

        # Define the Bidirectional RNN layer
        self.birnn = nn.LSTM(self.input_dim, self.att_rnn_size, self.num_layers,dropout=0.5,bidirectional=True)

        # Define the weight model
        self.weight_fc = nn.Linear(2*att_rnn_size, 1)
        self.att_sig_w = 3
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        lstm_out, hidden = self.birnn(x)
        weights = self.weight_fc(lstm_out)
        weights = weights * self.att_sig_w
        weights = self.sigmoid(weights)

        return weights

# Here we define our model as a class
class TAGM(BaseModel):
    """ Top model of TAGM"""
    def __init__(self, input_dim, hidden_dim, if_attention, output_dim=2,
                    num_layers=2):
        super(TAGM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.if_attention = if_attention

        if if_attention:
            self.attention = Attention(self.input_dim, 32)

        # Define the LSTM layer
        # self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
        # self.tagm = TAGM_model(self.input_dim, self.hidden_dim).to("cuda:0")
        self.tagm = TAGM_model(self.input_dim, self.hidden_dim)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)
        self.criterion = nn.CrossEntropyLoss()


    def forward(self,x):
        self.hidden = torch.randn(1, self.hidden_dim)
        outputs=[]
        x = x.view(x.size(1),x.size(0),x.size(2))
        seqLen = x.size(0)

        if self.if_attention:
            weigths = self.attention(x)
        else:
            weigths = torch.ones(seqLen)

        for i in range(0,seqLen):
            self.hidden = self.tagm(x[i], weigths[i], self.hidden)

        y = self.linear(self.hidden)

        # # using the last output (TAGM strategy)
        #
        # outs = torch.randn(1,2,requires_grad=True)
        # outs.data = y.view(1,self.output_dim)

        return y

    def loss(self, outputs, targets):
        loss = self.criterion(outputs, targets)

        return loss
