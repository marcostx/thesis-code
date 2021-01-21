import os
import torch
import sys
import numpy as np
from factories.optimizer import get_optimizer_parameters

from collections import Counter
from train.train import train, qualitative_analysis
from train.kfold import kfold, kfold_qualitative

from factories.optimizer import OptimizerFactory
from factories.model import ModelFactory

from collections import OrderedDict
from utils.model_utils import get_model_params
from utils.dataset_utils import get_dataset_params

from datasets.RWF_dataset import RWFDataset

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

    if qualitative_flag:
        kfold_qualitative(device=device,
            n_epochs=n_epochs,
            optimizer_kwargs=optimizer_kwargs,
            batch_size=batch_size,
            model_params=model_params,
            dataset_params=dataset_params,
            metadata_path=metadata_path)
    else:        
        kfold(device=device,
            n_epochs=n_epochs,
            optimizer_kwargs=optimizer_kwargs,
            batch_size=batch_size,
            model_params=model_params,
            dataset_params=dataset_params,
            metadata_path=metadata_path)

if __name__ == "__main__":
    main()
    print("Done!")
