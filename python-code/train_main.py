import os
import torch
import sys
import numpy as np
from factories.optimizer import get_optimizer_parameters

from collections import Counter
from train.train import train, qualitative_analysis, test_mediaeval

from factories.optimizer import OptimizerFactory
from factories.dataset import DatasetFactory
from factories.model import ModelFactory

from collections import OrderedDict
from utils.model_utils import get_model_params

from utils.dataset_utils import get_dataset_params, get_splits, get_images, early_fusion

from datasets.RWF_dataset import RWFDataset
from utils.sampler import BalancedBatchSampler

from utils.spatial_transforms import (Compose, ToTensor, Scale, Normalize, MultiScaleCornerCrop,
                                      RandomHorizontalFlip, CenterCrop)

from torchvision import transforms
import h5py
from utils.args import get_args
from utils.experiment import init_experiment, ini_checkpoint_experiment
from sklearn.model_selection import train_test_split
from torchvision import models

from utils.slack_message import send_msg

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

os.environ["HOSTNAME"] = "Macbook do Marcos"
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ['TORCH_HOME'] = 'cache2/'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def main():
    args = get_args()

    # descriptor = models.resnet101(pretrained=True)

    # torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    # Logs vars
    metadata_path = 'metadata'

    # raw data / or using feature extraction
    use_raw = args.use_raw

    # Dataset path
    data_path = args.datapath

    # feature extractor
    feat_model = args.feature_model

    # qualitative flag
    qualitative_flag = args.qualitative

    # fusion method (when used)
    fusion_method = args.fusion_method

    # Train varss
    ini_epoch = 0
    n_epochs = args.n_epochs
    dropout = args.dropout_prob
    lr = args.lr
    wd = args.wd
    opt_name = args.opt_name
    n_frames = args.n_frames
    batch_size = args.batch_size

    # Network vars
    dropout_prob = args.dropout_prob

    # Get optimizer vars
    optimizer_kwargs = get_optimizer_parameters(opt_name, lr=lr, wd=wd)

    # Dataset vars
    dataset_name = args.dataset_name

    dataset_params = get_dataset_params(dataset_name)

    model_name = args.model_name
    pretrained = args.pretrained

    model_params = get_model_params(args.model_name,
                                    dataset_name,
                                    pretrained=pretrained,
                                    hidden_size=args.hidden_size,
                                    feature_dim=args.feature_dim,
                                    att_rnn_size=args.att_rnn_size,
                                    att_type=args.att_type,
                                    if_att=args.if_att,
                                    bh=False,
                                    dropout_prob=dropout_prob)

    # dataset_type = get_dataset_by_model(model_name, dataset_name)

    # Creating model
    model = ModelFactory.factory(**model_params)

    model = model.cuda()

    # Data loading
    # Train set
    train_set, test_y = DatasetFactory.factory(args, test=False)
    if dataset_name == 'mediaeval':
        trainLoader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, pin_memory=True, sampler=BalancedBatchSampler(train_set, test_y),
                                                  num_workers=2)
    else:
        trainLoader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                                  num_workers=2)
    # Test set
    test_set, test_y = DatasetFactory.factory(args, test=True)

    if dataset_name == 'mediaeval':
        testLoader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True,
                                                 num_workers=2)
    else:
        testLoader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True,
                                                 num_workers=2)

    # video used for qualitative analysis
    if qualitative_flag:
        # video_selected_index = 131
        testLoader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                                 shuffle=False, num_workers=2)

        check = torch.load('checkpoint.pth')
        # new_state_dict = OrderedDict()
        # for k, v in check.items():
        #     # tirando o prefixo 'descriptor.' das keys
        #     name = k[7:]
        #     new_state_dict[name] = v

        # model.load_state_dict(new_state_dict)
        model.load_state_dict(check)

        qualitative_analysis(model=model,
                             device=device,
                             testLoader=testLoader)
        exit(1)
    # Create optimizer
    optimizer = OptimizerFactory.factory(
        model.parameters(), **optimizer_kwargs)

    filename = args.checkpoint_path
    if filename != "":
        if os.path.isfile(filename):
            ini_checkpoint_experiment(filename, model_name, dataset_name)

            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            ini_epoch = checkpoint['epoch']

            if isinstance(model, torch.nn.DataParallel):
                model.module.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                filename, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(filename))
            exit()
    else:
        # Init experiment
        model_dir, log_dir, experiment_id = init_experiment(
            metadata_path, dataset_name, model_name)

    if dataset_name == 'mediaeval':
        print("Mediaeval experiments")
        train_metrics = train(model=model,
                              optimizer=optimizer,
                              ini_epoch=ini_epoch,
                              n_epochs=n_epochs,
                              device=device,
                              trainloader=trainLoader,
                              testloader=testLoader,
                              model_dir=model_dir,
                              log_dir=log_dir,
                              is_mediaeval=True,
                              test=False,
                              cv_idx=1)

        test_mediaeval(device, testLoader, model, model_name,
                       args.if_att, args.att_type)
        exit(1)
    else:
        train_metrics = train(model=model,
                              optimizer=optimizer,
                              ini_epoch=ini_epoch,
                              n_epochs=n_epochs,
                              device=device,
                              trainloader=trainLoader,
                              testloader=testLoader,
                              model_dir=model_dir,
                              log_dir=log_dir)
    print("valid acc:", train_metrics["test"]["acc"])
    print("Kfold finished log path:", log_dir)

    msg = """
    ```
    Exp Name: `{}``
    Host Machine: `{}`
    acc train: `{}`
    acc validation: `{}`
    log path: `{}`
    attention type: `{}`
    hidden size: `{}`
    batch size: `{}`
    learning rate: `{}`
    ```
    """.format(experiment_id, os.environ["HOSTNAME"],  train_metrics["train"]["acc"], train_metrics["test"]["acc"], log_dir, args.att_type, args.hidden_size, batch_size, lr)
    send_msg(msg)


if __name__ == "__main__":
    main()
    print("Done!")
