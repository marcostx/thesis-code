import torch
from torch import nn
import torch.nn.functional as F
from .cbam import *
from models.base_model import BaseModel
from torchvision import models
from collections import OrderedDict


def weight_model(att_rnn_size, att_sig_w):
    model = nn.Sequential(
        nn.Linear(2 * att_rnn_size, 1),
        nn.MulConstant(att_sig_w),
        nn.Sigmoid()
    )
    return model


class MultipleLayerAttention(nn.Module):

    def __init__(self, hidden_dim, attention_dim):
        super(MultipleLayerAttention, self).__init__()
        self.latent_attention = nn.Linear(hidden_dim, attention_dim)
        self.hidden_attention = nn.Linear(hidden_dim, attention_dim)
        self.joint_attention = nn.Linear(attention_dim, 1)

    def forward(self, hidden_repr):
        h_t = hidden_repr[0]
        latent_att = self.latent_attention(h_t)
        hidden_att = self.hidden_attention(h_t)
        joint_att = self.joint_attention(
            F.relu(latent_att + hidden_att)).squeeze(-1)
        attention_w = F.softmax(joint_att, dim=-1)
        return attention_w


class SoftAttention(BaseModel):
    """ Soft Attention Vanilla."""

    def __init__(self, input_dim):
        super(SoftAttention, self).__init__()
        self.net = nn.Linear(input_dim, 1)

    def forward(self, x):
        out = self.net(x)
        normalized_out = F.softmax(out, 1)
        return normalized_out

# Here we define our model as a class


class LSTM(BaseModel):

    def __init__(self, model_name, input_dim, att_type, hidden_dim, if_attention, dropout_prob=0.5, output_dim=2,
                 num_layers=1):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.if_att = if_attention
        self.att_type = att_type
        self.dropout_prob = dropout_prob

        self.attention_hidden_dim = int(self.hidden_dim / 2)

        # # if fineutuning ==True
        # print("====================================")
        # print("Feature Extraction VGG19: finetuning")
        # print("====================================")
        # descriptor = models.vgg19(pretrained=True)
        # descriptor.classifier[6] = nn.Linear(4096, 2)

        # check = torch.load('checkpoint-vgg19.pt')
        # state_dicts = check
        # new_state_dict = OrderedDict()
        # for k, v in state_dicts.items():
        #     # tirando o prefixo 'descriptor.' das keys
        #     name = k[18:]
        #     new_state_dict[name] = v

        # descriptor.load_state_dict(new_state_dict)
        # modules=list(descriptor.classifier.children())[:-4]
        # descriptor.classifier = nn.Sequential(*modules)
        # self.feature_extractor = descriptor
        # for param in self.feature_extractor.parameters():
        #     param.requires_grad = False

        if self.if_att:
            if self.att_type == "soft_att":
                print("ATT OPTION : Model attention")
                self.attention = SoftAttention(self.hidden_dim * 2)
            elif self.att_type == "ml_att":
                print("ATT OPTION : Multiple Layer attention")
                self.attention = MultipleLayerAttention(
                    self.hidden_dim, self.attention_hidden_dim)
            elif self.att_type == "uniform":
                print("ATT OPTION : Uniform weighting")
        else:
            print("LAST SEGMENT")

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers,
                            batch_first=True, dropout=self.dropout_prob, bidirectional=True)

        # # init the network
        # for name, param in self.lstm.named_parameters():
        #   if 'bias' in name:
        #      nn.init.constant(param, 1.0)
        #   elif 'weight' in name:
        #      nn.init.xavier_normal(param)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim * 2, output_dim)
        # self.linear.apply(weight_init)

        # self.hidden = self.init_hidden()
        self.criterion = nn.CrossEntropyLoss().cuda()

    # def init_hidden(self):
    #     h0 = Variable(torch.zeros(2, 1, self.hidden_dim).cuda())
    #     c0 = Variable(torch.zeros(2, 1, self.hidden_dim).cuda())
    #
    #     return (h0, c0)

    def forward(self, x):
        # features = []
        # for idx,batch_item in enumerate(x):
        #     feature = self.feature_extractor(batch_item)
        #     features.append(feature)
        # x = torch.stack(features)

        outputs = []
        weights = None
        # hidden = torch.randn(1, 1, self.hidden_dim).cuda()

        lstm_out, hidden = self.lstm(x, None)

        if self.if_att: 
            if self.att_type == "soft_att":
                weights = self.attention(lstm_out)
                lstm_out = torch.mul(lstm_out, weights)
                lstm_out = lstm_out.sum(1)
            elif self.att_type == 'ml_att':
                lstm_out = torch.sum(self.attention(
                    lstm_out).unsqueeze(-1) * lstm_out, dim=0)
            elif self.att_type == "uniform":
                atts = torch.full((lstm_out.size(0), 1, 1),
                                  (1.0 / lstm_out.size(0))).cuda()
                lstm_out = torch.mul(lstm_out, atts)
                lstm_out = lstm_out.sum(1)
        # lstm_mean_out = lstm_out.mean(0)

        if not self.if_att:
            y = self.linear(lstm_out[:, -1, :])
        else:
            y = self.linear(lstm_out)
        y = F.softmax(y)

        # y = F.log_softmax(y, dim=1)
        # using the last output (TAGM strategy)

        # outs = torch.randn(1,2,requires_grad=True)
        # outs.data = y.view(1,self.output_dim)
        return y, weights

    def loss(self, outputs, targets):
        loss = self.criterion(outputs, targets)

        return loss
