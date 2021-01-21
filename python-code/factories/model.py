from __future__ import absolute_import, division, print_function

import torch

from models.convLSTM import *
from models.resnetCBAM import *
from models.lstm import *
from models.lstmLateFusion import *
from models.tagm import *
from models.attentionClusters import *
from models.vgg import *
from models.multimodalAttentionClusters import *
from models.attentionClustersLateFusion import *


class ModelFactory(object):
    """Model factory return instance of model specified in type."""
    @staticmethod
    def factory(*args, **kwargs):
        # print(kwargs)
        if kwargs["model_name"] == "convLSTM":
            model = ConvLSTM(kwargs["hidden_dim"])
        elif kwargs["model_name"] == "cbam":
            model = ResidualNet(18, 2, 'CBAM')
        elif kwargs["model_name"] == "lstm":
            model = LSTM(*args, **kwargs)
        elif kwargs["model_name"] == "lstm_late":
            model = LSTMLateFusion(*args, **kwargs)
        elif kwargs["model_name"] == "vgg":
            print("finetuning : on")
            model = VGG19(kwargs['input_dim'])
        elif kwargs["model_name"] == "att_clusters":
            model = AttentionClusters(kwargs['input_dim'])
        elif kwargs["model_name"] == "att_clusters_multimodal":
            model = AttentionClustersMultimodal(kwargs['input_dim'])
        elif kwargs["model_name"] == "att_clusters_late":
            model = AttentionClustersLateFusion(kwargs['input_dim'])

        else:
            assert 0, "Bad model_name of model creation: " + \
                kwargs["model_name"]

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")

            model = torch.nn.DataParallel(model)

        return model
