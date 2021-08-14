from __future__ import absolute_import, division, print_function

import torchvision
import h5py
import torch
import numpy as np
from collections import Counter

from datasets.RWF_dataset import RWFDataset
from sklearn.model_selection import train_test_split

from utils.spatial_transforms import (Compose, ToTensor, Scale, Normalize, MultiScaleCornerCrop,
                                      RandomHorizontalFlip, CenterCrop)

from utils.sampler import BalancedBatchSampler


class DatasetFactory(object):
    """Dataset factory return instance of dataset specified in dataset_type."""
    @staticmethod
    def factory(args, test=True):
        model_name = args.model_name
        dataset_name = args.dataset_name

        if dataset_name == 'rwf':
            train_h5 = h5py.File(
                '/home/src/data/rwf-train-efficientnet-fine.h5', 'r')
            test_h5 = h5py.File(
                '/home/src/data/rwf-test-efficientnet-fine.h5', 'r')
        elif dataset_name == 'mediaeval':
            train_h5 = h5py.File(
                '/home/src/data/mediaeval-train-efficientnet.h5', 'r')
            test_h5 = h5py.File(
                '/home/src/data/mediaeval-test-efficientnet.h5', 'r')

        trainx, trainy = train_h5['x'], train_h5['y']
        testx, testy = test_h5['x'], test_h5['y']

        if model_name == 'att_clusters_multimodal' or 'late' in model_name:
            if model_name == 'att_clusters_multimodal':
                print("Fusion method : Attention Clusters fusion")
            elif model_name == 'att_clusters_late':
                print("Fusion method : Late fusion [Attention Clusters]")
            elif model_name == 'lstm_late':
                print("Fusion method : Late fusion [Bidirectional LSTM]")
            if dataset_name == 'rwf':
                train_h5_flow = h5py.File(
                    '/home/src/data/rwf-train-efficientnet-flow.h5', 'r')
                test_h5_flow = h5py.File(
                    '/home/src/data/rwf-test-efficientnet-flow.h5', 'r')
            elif dataset_name == 'mediaeval':
                train_h5_flow = h5py.File(
                    '/home/src/data/mediaeval-train-efficientnet-flow.h5', 'r')
                test_h5_flow = h5py.File(
                    '/home/src/data/mediaeval-test-efficientnet-flow-filtered.h5', 'r')

            trainx_f, trainy_f = train_h5_flow['x'], train_h5_flow['y']
            testx_f, testy_f = test_h5_flow['x'], test_h5_flow['y']

            trainx_ = np.zeros(
                (trainx_f.shape[0], 2*trainx_f.shape[1], args.feature_dim))
            testx_ = np.zeros(
                (testy_f.shape[0], 2*trainx_f.shape[1], args.feature_dim))

            trainx_[:, :trainx_f.shape[1], :] = trainx
            trainx_[:, trainx_f.shape[1]:, :] = trainx_f

            testx_[:, :trainx_f.shape[1], :] = testx
            testx_[:, trainx_f.shape[1]:, :] = testx_f

            trainx = trainx_
            testx = testx_

        if test:
            test_spatial_transform = Compose([ToTensor()])

            vidSeqValid = RWFDataset(
                testx, testy, test_spatial_transform, 'valid', False)

            return vidSeqValid, testy
        else:
            spatial_transform = Compose([ToTensor()])

            vidSeqTrain = RWFDataset(
                trainx, trainy, spatial_transform, 'train', False)
            return vidSeqTrain, trainy
