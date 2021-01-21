import os
import torch
import numpy as np
from torch.utils.data import Dataset

from torchvision import transforms, models
from torch import nn
import glob
import h5py
from tqdm import tqdm
from keras.preprocessing import image

os.environ['TORCH_HOME'] = 'cache2/'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

from datasets.base_dataset import BaseDataset


class CCTVDataset(BaseDataset):
    """CCTVDataset dataset"""

    def __init__(self, dataset, labels, feat_model, spatial_transform, n_frames=150, split='train', use_raw=True):
        self.spatial_transform = spatial_transform
        self.images = dataset
        self.labels = labels
        self.n_frames = n_frames
        self.feat_model = feat_model
        self.split = split
        self.use_raw = use_raw

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_raw:
            vid_name = self.images[idx]
            label = self.labels[idx]
            videoImgs = glob.glob(vid_name + "/*.png")
            #
            inpSeq = []

            self.spatial_transform.randomize_parameters()
            for id_ in range(0, len(videoImgs)):
                img = videoImgs[id_]

                # img = image.load_img(img, target_size=(224, 224))
                # x = image.img_to_array(img)
                # x = np.expand_dims(x, axis=0)
                # x = x.reshape(x.shape[0],x.shape[3],x.shape[2],x.shape[1])
                img = image.load_img(img)

                inpSeq.append(self.spatial_transform(img.convert('RGB')))

            inpSeq = torch.stack(inpSeq, 0)
        else:
            data = self.images[idx]
            label = self.labels[idx]
            inpSeq = torch.tensor(data)
            #
            inpSeq = inpSeq.resize_(inpSeq.size(1), 1, inpSeq.size(0))

        return inpSeq, label
