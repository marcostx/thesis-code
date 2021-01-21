import os
import glob
import torch
import numpy as np
from os.path import join
import numpy as np
import pandas as pd
from PIL import Image
from skimage import io
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision import models
from torch import nn
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input

os.environ['TORCH_HOME'] = 'cache2/'

from tqdm import tqdm

from datasets.base_dataset import BaseDataset


class ViolentFlowsDataset(BaseDataset):
    """Violentflows dataset"""

    def __init__(self, dataset, labels, spatial_transform, n_frames=30):
        self.spatial_transform = spatial_transform
        self.images = dataset
        self.labels = labels
        self.n_frames = n_frames

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        vid_name = self.images[idx]
        label = self.labels[idx]

        if os.path.exists(vid_name + "violent"):
            vid_name = vid_name + "violent"
        else:
            vid_name = vid_name + "nonviolent"

        video_figures = glob.glob(vid_name + "/*.jpg") + \
            glob.glob(vid_name + '/*.png')

        inpSeq = []
        self.spatial_transform.randomize_parameters()
        for vid in video_figures:
            img = Image.open(vid)
            inpSeq.append(self.spatial_transform(img.convert('RGB')))

        inpSeq = torch.stack(inpSeq, 0)
        return inpSeq, label


class ViolentFlowsDatasetVGG(Dataset):
    """Violentflows dataset with vgg19 as feature extraction"""

    def __init__(self, data_dir, labels, n_frames, spatial_transform):
        self.spatial_transform = spatial_transform
        self.images = data_dir
        self.labels = labels
        self.n_frames = n_frames
        self.class_size = 2
        self.classes = ['nonviolence', 'violence']
        self.feature_dim = 4096
        #
        # basemodel = models.resnet101(pretrained=True)
        # modules=list(basemodel.children())[:-1]
        # resnet=nn.Sequential(*modules)
        # self.feature_extraction = resnet
        # for i, param in enumerate(self.feature_extraction.parameters()):
        #     param.requires_grad = False
        # print(self.feature_extraction)

        # basemodel = models.vgg19(pretrained=True)
        # modules=list(basemodel.classifier.children())[:-4]
        # basemodel.classifier=nn.Sequential(*modules)
        # self.feature_extraction = basemodel
        # for i, param in enumerate(self.feature_extraction.parameters()):
        #     param.requires_grad = False
        #
        # self.make_dataset()

    def make_dataset(self):
        """ extracting features and saving"""
        len_vids = []
        self.feature_extraction.eval()

        for idx in range(len(self.images)):
            vid_name = self.images[idx]
            if os.path.exists(vid_name + "violent"):
                vid_name = vid_name + "violent"
            else:
                vid_name = vid_name + "nonviolent"

            imgs = glob.glob(vid_name + '/*.jpg') + \
                glob.glob(vid_name + '/*.png')
            imgs = sorted(imgs)

            for i in range(int((len(imgs) / 2)) - int(self.n_frames / 2), int((len(imgs) / 2)) + int(self.n_frames / 2) + 1):
                if i > len(imgs) - 1:
                    break
                if not os.path.exists(imgs[i]):
                    break

                # if os.path.exists(vid_name+'/'+os.path.basename(imgs[i]).split('.')[0]+'.pt'):
                #     print("Skipping...")
                #     continue
                fl_name = imgs[i]
                img = image.load_img(fl_name, target_size=(224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                # x = preprocess_input(x)
                x = x.reshape(x.shape[0], x.shape[3], x.shape[2], x.shape[1])

                # shape (1, 4096)
                vgg_features = self.feature_extraction(torch.Tensor(x))

                torch.save(vgg_features, vid_name + '/' +
                           os.path.basename(fl_name).split('.')[0] + '_resnet.pt')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        vid_name = self.images[idx]
        label = self.labels[idx]
        # inpSeq=[]

        # cuda0 = torch.device('cuda:0')
        inpSeq = torch.zeros([self.n_frames, self.feature_dim])
        if "violence" in vid_name:
            label = 1
        else:
            label = 0

        tensors = sorted(glob.glob(vid_name + '/*.pt'))
        tensors = sorted(list(filter(lambda x: "_resnet" not in x, tensors)))
        for i in range(0, self.n_frames):
            pt = torch.load(tensors[i])

            inpSeq[i] = pt
            # inpSeq.append(pt.view(4096))
        # inpSeq = torch.stack(inpSeq, 0)

        return inpSeq, label
