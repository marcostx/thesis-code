import torch
import os
#
# def get_dataset_by_model(model_name, dataset_name, *args, **kwargs):
#     if dataset_name in ["UCF-101"]:
#         return 'UCF-101'
#     if model_name in ["c3d", "c3d_top", "c3d_lstm", "cnn3d", "cnn3d_lstm", "inceptionv3_lstm", "vgg16_lstm", "resnet3d_101","resnet3d_18", "i3d", "resnet101_lstm", "resnet18_lstm"]:
#         return 'video'
#     elif model_name in ["c3d_block_lstm", "cnn3d_block_lstm", "crnn", "crnn_nf", "crnn_simple_sum", "crnn_skip", "crnn_attention", "crnn_attention_nn"]:
#         return "blocks"


def get_model_params(model_name, dataset_name, att_rnn_size,att_type, hidden_size,feature_dim, if_att, pretrained, bh=False, dropout_prob=0.5):
    n_gpus = torch.cuda.device_count() if torch.cuda.device_count() > 0 else 1

    img_size = 100
    n_frames = 20
    n_blocks = 5
    frames_per_block = 5
    model_kwargs = {}

    if model_name == "inceptionv3_lstm":
        batch_size = 3
        img_size = 300
        model_kwargs = {}

    elif  model_name == "vgg16_lstm":
        batch_size = 3
        img_size = 224
        model_kwargs = {}

    elif model_name == "resnet101_lstm":
        batch_size = 3
        img_size = 224
        model_kwargs = {}

    elif model_name == "resnet18_lstm":
        img_size = 224
        model_kwargs = {}

    elif model_name == "c3d":
        n_frames = 16
        n_blocks = 4
        frames_per_block = 4
        model_kwargs = {}

    elif model_name == "resnet3d_101":
        model_kwargs = {}

    elif model_name == "resnet3d_18":
        batch_size = 20
        model_kwargs = {}

    elif model_name == "i3d":
        batch_size = 5
        img_size = 226
        n_frames = 64
        n_blocks = 8
        frames_per_block = 8
        model_kwargs = {
            "dataset_name": dataset_name
        }

    elif model_name == "c3d_block_lstm":
        model_kwargs = {}

    elif model_name == "c3d_top":
        batch_size = 10
        model_kwargs = {
            "bh": bh
        }


    model_kwargs = {
        "model_name": model_name,
        "if_attention": if_att,
        "att_type": att_type,
        "hidden_dim": hidden_size,
        "input_dim": feature_dim,
        "dropout_prob": dropout_prob
    }
    # batch_size =  batch_size * n_gpus

    # return model_kwargs, batch_size
    return model_kwargs
