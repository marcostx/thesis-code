import os
import torch
import numpy as np

import cv2
import glob
from PIL import Image
from keras.preprocessing import image

os.environ['TORCH_HOME'] = 'cache2/'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

from datasets.base_dataset import BaseDataset
import sys

class RWFDataset(BaseDataset):
    """RWFDataset dataset"""

    def __init__(self, dataset, labels, spatial_transform, split='train', use_raw=True):
        self.spatial_transform = spatial_transform
        self.images = dataset
        self.labels = labels
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
            for id_ in range(0, len(videoImgs) - 1):
                # -- --- ---- ---------- ---- --- --
                # -- --- ---- RGB SCHEME ---- --- --
                # -- --- ---- ---------- ---- --- --
                # img = videoImgs[id_]
                # img = image.load_img(img)
                #
                # inpSeq.append(self.spatial_transform(img.convert('RGB')))

                # -- --- ---- ---------- ---- --- --
                # -- --- ----  OPT FLOW  ---- --- --
                # -- --- ---- ---------- ---- --- --

                img1 = cv2.imread(videoImgs[id_])
                img2 = cv2.imread(videoImgs[id_ + 1])
                img1 = cv2.resize(img1, (224, 224),
                                  interpolation=cv2.INTER_AREA)
                img2 = cv2.resize(img2, (224, 224),
                                  interpolation=cv2.INTER_AREA)
                img1_ = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                img2_ = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                img1_ = np.reshape(img1_, (224, 224, 3))
                img2_ = np.reshape(img2_, (224, 224, 3))
                img1_ = cv2.cvtColor(img1_, cv2.COLOR_RGB2GRAY)
                img2_ = cv2.cvtColor(img2_, cv2.COLOR_RGB2GRAY)
                img1_ = np.reshape(img1_, (224, 224, 1))
                img2_ = np.reshape(img2_, (224, 224, 1))

                flow = cv2.calcOpticalFlowFarneback(
                    img1_, img2_, None, 0.5, 3, 15, 3, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

                # converting to RGB
                hsv = np.zeros_like(img1)
                hsv[..., 1] = 255

                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

                # Add into list
                x = Image.fromarray(rgb)
                x = self.spatial_transform(x.convert("RGB"))
                inpSeq.append(x)

            # Padding the last frame as empty array
            # x = Image.new('RGB', (224,224))
            # x = self.spatial_transform(x.convert("RGB"))
            # inpSeq.append(x)

            inpSeq = torch.stack(inpSeq, 0)
        else:
            # from h5 file
            data = self.images[idx]
            label = self.labels[idx]
            inpSeq = torch.tensor(data)
            #
            inpSeq = inpSeq.resize_(inpSeq.size(0), inpSeq.size(1))

        return inpSeq, label
