import torch
import os
import numpy as np

from datasets.RWF_dataset import RWFDataset

from utils.spatial_transforms import (Compose, ToTensor, Scale, Normalize, MultiScaleCornerCrop,
                                      RandomHorizontalFlip, CenterCrop)

from factories.optimizer import OptimizerFactory
from factories.model import ModelFactory

from train.train import train, qualitative_analysis
from utils.slack_message import send_msg

from sklearn.model_selection import StratifiedKFold
from utils.experiment import init_experiment, init_kfold_subexperiment, ini_kfold_checkpoint_experiment
import h5py
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def load_data(model_name):

    train_h5 = h5py.File('/home/src/hockey-efficientnet.h5', 'r')
    # #
    trainx, trainy = train_h5['x'], train_h5['y']

    if model_name == 'att_clusters_multimodal' or 'late' in model_name:
        if model_name == 'att_clusters_multimodal':
            print("Fusion method : Attention Clusters fusion")
        elif model_name == 'att_clusters_late':
            print("Fusion method : Late fusion [Attention Clusters]")
        elif model_name == 'lstm_late':
            print("Fusion method : Late fusion [Bidirectional LSTM]")
        train_h5_flow = h5py.File('hockey-flow-efficientnet.h5', 'r')

        trainx_f = train_h5_flow['x']

        trainx_ = np.zeros((trainx_f.shape[0], 80, 1536))

        trainx_[:, :40, :] = trainx
        trainx_[:, 40:, :] = trainx_f

        trainx = trainx_

    return trainx, trainy

def kfold_qualitative(device, n_epochs, optimizer_kwargs, batch_size,
          model_params, dataset_params, metadata_path):
    dataset_name = dataset_params["dataset_name"]
    model_name = model_params["model_name"]

    k = dataset_params['k']
    i = 0

    # Init experiment
    test_acc_list = {}
    models_dir, logs_dir, experiment_id = init_experiment(metadata_path, dataset_name, model_name)
 
    # whole dataset
    X, y = load_data(model_name) 

    # get splits
    skf = StratifiedKFold(n_splits=k)

    for train_index, test_index in skf.split(X, y):
        trainx, testx = X[train_index], X[test_index]
        trainy, testy = y[train_index], y[test_index]

        # Load model
        model = ModelFactory.factory(**model_params)
        model.cuda()

        spatial_transform = Compose([ToTensor()])

        vid_seq_Valid = RWFDataset(testx, testy, spatial_transform=spatial_transform,split='valid', use_raw=False)

        test_loader = torch.utils.data.DataLoader(vid_seq_Valid, batch_size=1,
                                                shuffle=False, num_workers=2)

        check = torch.load('checkpoint-1.pt')
        model.load_state_dict(check)

        qualitative_analysis(model=model,
                            device=device,
                                testLoader=test_loader)
        i+=1

def kfold(device, n_epochs, optimizer_kwargs, batch_size,
          model_params, dataset_params, metadata_path):
    dataset_name = dataset_params["dataset_name"]
    model_name = model_params["model_name"]

    k = dataset_params['k']
    i = 0


    # Init experiment
    test_acc_list = {}
    models_dir, logs_dir, experiment_id = init_experiment(metadata_path, dataset_name, model_name)
 
    # whole dataset
    X, y = load_data(model_name) 

    # get splits
    skf = StratifiedKFold(n_splits=k)

    for train_index, test_index in skf.split(X, y):

        trainx, testx = X[train_index], X[test_index]
        trainy, testy = y[train_index], y[test_index]

        # Load model
        model = ModelFactory.factory(**model_params)
        model.cuda()

        print(model.parameters)

        spatial_transform = Compose([ToTensor()])

        vid_seq_train = RWFDataset(trainx, trainy, spatial_transform=spatial_transform, split='train', use_raw=False)

        train_loader = torch.utils.data.DataLoader(vid_seq_train, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)

        vid_seq_Valid = RWFDataset(testx, testy, spatial_transform=spatial_transform,split='valid', use_raw=False)

        test_loader = torch.utils.data.DataLoader(vid_seq_Valid, batch_size=batch_size,
                                                 shuffle=True, num_workers=2)

        # Create optimizer
        optimizer = OptimizerFactory.factory(model.parameters(), **optimizer_kwargs)

        # Init experiment
        model_dir, log_dir, kfold_experiment_id = init_kfold_subexperiment(models_dir, logs_dir, i, experiment_id)

        train_metrics = train(model=model,
                              optimizer=optimizer,
                              n_epochs=n_epochs,
                              device=device,
                              trainloader=train_loader,
                              testloader=test_loader,
                              model_dir=model_dir,
                              log_dir=log_dir,
                              cv_idx=i+1)

        if "test" in train_metrics and "acc" in train_metrics["test"]:
            test_acc_list[i] = train_metrics["test"]["acc"]
        i+=1

    test_list = np.zeros(k,dtype=np.float)
    for k, v in test_acc_list.items():
        test_list[k] = v
    test_list = list(test_list)

    print("test acc list:", test_list)
    print("kfold test acc:", np.mean(test_list))

    print("Kfold finished log path:", logs_dir)