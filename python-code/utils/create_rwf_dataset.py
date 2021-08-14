import sys
import cv2
import time
import h5py
import argparse
import subprocess
import numpy as np
from collections import OrderedDict
from tensorflow.keras.models import load_model, Sequential, Model
from efficientnet.tfkeras import center_crop_and_resize, preprocess_input
from efficientnet.tfkeras import EfficientNetB3
from os.path import basename, join, exists, splitext, dirname
from torchvision import transforms, models
# from spatial_transforms import (Compose, ToTensor, FiveCrops, Scale, Normalize, MultiScaleCornerCrop,
#                                RandomHorizontalFlip, TenCrops, FlippedImagesTest, CenterCrop)
import glob
# import tensorflow as tf
import torch
from torch import nn
from PIL import Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# from keras.preprocessing import image


VIDEO_EXTENSION = "*.avi"
IMG_EXTENSION = ".png"
AUDIO_EXTENSION = ".wav"
os.environ['TORCH_HOME'] = 'cache2/'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def getArgs():
    argparser = argparse.ArgumentParser(description=__doc__)

    #argparser.add_argument('-dp','--datapath', default='/Users/marcostexeira/pay-attention-pytorch/data/raw_frames/violentflow', help='Directory containing data sequences', type=str)
    argparser.add_argument('-dp', '--datapath', default='/home/datasets/',
                           help='Directory containing data sequences', type=str)
    argparser.add_argument('-dn', '--dataset_name',
                           default='rwf', help='dataset name', type=str)
    argparser.add_argument('-m', '--modality', default='flow',
                           help='optical flow / raw frames', type=str)
    argparser.add_argument('-v', '--violence_mediaeval', default='violence_test.txt',
                           help='violent videos list', type=str)
    argparser.add_argument('-sn', '--split_name',
                           default='train', help='(train/val/test)', type=str)
    argparser.add_argument('-ou', '--output_path', default='/Users/marcostexeira/masters-project/ucf-crime/',
                           help='Directory containing data sequences', type=str)
    argparser.add_argument('-g', '--gpu', default=True,
                           help='use gpu or not', type=bool)
    argparser.add_argument(
        '-l', '--library', default='keras', help='keras/pytorch', type=str)
    argparser.add_argument('-f', '--finetuning', default=True,
                           help='load model finedtuned or not', type=bool)
    argparser.add_argument('-s', '--st', default=0,
                           help='index start video', type=int)
    argparser.add_argument('-e', '--end', default=800,
                           help='index end video', type=int)

    args = argparser.parse_args()
    return args


def train_violence_parser(file_inp, video_files):
    file_inp = open(file_inp)
    lines = file_inp.readlines()
    lines = [str.replace("\n", "") for str in lines]

    video_files_wt_ext = [basename(video).split(".")[0]
                          for video in video_files]

    y = list(filter(lambda video: video in lines, video_files_wt_ext))

    return y


def test_violence_parser(file_inp, video_files):
    file_inp = open(file_inp)
    lines = file_inp.readlines()
    lines = [str.replace("\n", "") for str in lines]
    violence = []
    for line in lines:
        if line.split(" ")[3] == "1":
            violence.append(line.split(" ")[2])

    video_files_wt_ext = [basename(video).split(".")[0]
                          for video in video_files]

    y = list(filter(lambda video: video in violence, video_files_wt_ext))

    return y


def extractFeatures(feature_extractor, video, fps, modality, dataset_name, only_frames=False):

    # cmd = "ffmpeg  -i {} -r {} -vf scale=640:-1 {}"
    cmd = "ffmpeg  -i {} -r {} -vf scale=360:-1 {}"
    cmd_delete = "rm -rf {}"
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # cmd_delete = "rm -R {}"

    video_features_arr = []
    dimmension_size = 300

    imgsFolder = join(dirname(video), splitext(basename(video))[0])
    if not exists(imgsFolder):
        os.mkdir(imgsFolder)
        subprocess.call(cmd.format(video, fps, join(
            imgsFolder, 'frame_%03d' + IMG_EXTENSION)), shell=True)
    else:
        print("Skipping ...")

    video_features = None

    imgs = glob.glob(join(imgsFolder, "*.png"))

    if only_frames or len(imgs) == 0:
        print("Discarding ...")
        return None
    if dataset_name == 'hockey':
        # minimum video size (hockey dataset)
        imgs = imgs[:40]
    elif dataset_name == 'mediaeval':
        # middle of the video considering a fixed sequence length
        video_seq = 60
        imgs = imgs[int((len(imgs)/2)-(video_seq)/2):int((len(imgs)/2)+(video_seq)/2)]

    start = time.time()
    xs = np.zeros((len(imgs), 300, 300, 3))
    if modality == 'raw':
        # for id_ in range(0, len(imgs)):
        for id_ in range(0, len(imgs)):
            img = imgs[id_]

            # EfficientNet as feature extractor
            inp_img = cv2.imread(img)
            inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB)

            inp_img = inp_img[:, :, :3]
            x = center_crop_and_resize(inp_img, image_size=300)
            # x = preprocess_input(x)
            x = np.expand_dims(x, 0)
            xs[id_, :, :, :] = x

            # print(x.shape)
            # features = feature_extractor.predict(x, batch_size=)
            # print(features.shape)

            # video_features = features if video_features is None else np.concatenate(
            #     [video_features, features], axis=0)
        video_features = feature_extractor.predict(xs, batch_size=150)

        if dataset_name == 'mediaeval':
            print("removendo frames ..")
            subprocess.call(cmd_delete.format(imgsFolder), shell=True)

    elif modality == 'flow':
        print("Extracting Optical Flow features ...")
        start = time.time()

        for idx_ in range(0, len(imgs) - 1):
            img1 = cv2.imread(imgs[idx_])
            img2 = cv2.imread(imgs[idx_ + 1])
            img1 = cv2.resize(
                img1, (dimmension_size, dimmension_size), interpolation=cv2.INTER_AREA)
            img2 = cv2.resize(
                img2, (dimmension_size, dimmension_size), interpolation=cv2.INTER_AREA)
            #img1_ = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            #img2_ = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            img1_ = np.reshape(img1, (dimmension_size, dimmension_size, 3))
            img2_ = np.reshape(img2, (dimmension_size, dimmension_size, 3))
            img1_ = cv2.cvtColor(img1_, cv2.COLOR_BGR2GRAY)
            img2_ = cv2.cvtColor(img2_, cv2.COLOR_BGR2GRAY)
            img1_ = np.reshape(img1_, (dimmension_size, dimmension_size, 1))
            img2_ = np.reshape(img2_, (dimmension_size, dimmension_size, 1))

            flow = cv2.calcOpticalFlowFarneback(
                img1_, img2_, None, 0.5, 3, 15, 3, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

            # converting to RGB
            hsv = np.zeros_like(img1)
            hsv[..., 1] = 255

            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            # efficient net as feature extractor
            x = center_crop_and_resize(rgb, image_size=300)
            x = preprocess_input(x)
            x = np.expand_dims(x, 0)
            xs[idx_, :, :, :] = x

            # features = feature_extractor.predict(x)

            # video_features = features if video_features is None else np.concatenate(
            #     [video_features, features], axis=0)

        if np.count_nonzero(xs) == 0:
            return None
        # Padding the last frame as empty array
        x = np.zeros((300, 300, 3), dtype=int)

        # x = preprocess_input(x)
        x = np.expand_dims(x, 0)

        if dataset_name == 'mediaeval':
            print("removendo frames ..")
            subprocess.call(cmd_delete.format(imgsFolder), shell=True)

        video_features = feature_extractor.predict(xs, batch_size=150)

        # video_features = features if video_features is None else np.concatenate(
        #     [video_features, features], axis=0)

    print("Time spent: {}".format(time.time() - start))

    # subprocess.call(cmd_delete.format(imgsFolder), shell=True)

    video_features_arr.append(video_features)

    video_features_arr = np.array(video_features_arr, dtype=np.float)

    return video_features_arr


def main():
    args = getArgs()

    modality = args.modality
    datasetName = args.dataset_name
    datasetPath = args.datapath
    split_name = args.split_name
    finetune = args.finetuning
    violence_mediaeval = args.violence_mediaeval

    start_v = args.st
    end_v = args.end
    print(start_v, end_v)

    positiveFolder = "Fight"
    negativeFolder = "NonFight"

    labels = []
    fps = 30
    feature_vector = None
    if datasetName == 'rwf':
        number_of_frames = 150
    elif datasetName == 'hockey':
        number_of_frames = 40
    elif datasetName == 'mediaeval':
        number_of_frames = 60

    start = time.time()

    if finetune:
        print("Loading finetuned model ...")
        model = load_model("/home/src/finetuned_efficientnet_rwf_flow.h5")
        # feature_extractor = Sequential()
        # for layer in model.layers[:-1]:
        #     print(layer)
        #     feature_extractor.add(layer)
        feature_extractor = Model(
            inputs=model.layers[0].input, outputs=model.layers[-2].output)
    else:
        feature_extractor = EfficientNetB3(
            weights='imagenet', include_top=False, pooling='avg')

    h5 = h5py.File('{}-{}-{}-{}.h5'.format(datasetName,
                                           split_name, start_v, end_v), 'w')
    # mediaevaltest_names = open("mediaevaltest_names_{}_{}.txt".format(start_v, end_v), 'a')

    if datasetName == 'rwf':
        # violent videos
        posVideoList = glob.glob(
            join(datasetPath, split_name, positiveFolder, VIDEO_EXTENSION))
        # non violent videos
        negVideoList = glob.glob(
            join(datasetPath, split_name, negativeFolder, VIDEO_EXTENSION))
    elif datasetName == 'hockey':
        posVideoList = glob.glob(
            join(datasetPath, positiveFolder, VIDEO_EXTENSION))
        negVideoList = glob.glob(
            join(datasetPath, negativeFolder, VIDEO_EXTENSION))
    elif datasetName == 'mediaeval':
        videos = glob.glob(
            join(datasetPath, split_name, VIDEO_EXTENSION))
        ignore_files_with_words = ["flip", "transpose"]
        videos = [x for x in videos if
                  all(y not in x for y in ignore_files_with_words)]

        if split_name == 'train':
            posVideoList = train_violence_parser(violence_mediaeval, videos)
        else:
            posVideoList = test_violence_parser(violence_mediaeval, videos)

        posVideoList = [join(datasetPath, split_name, video) +
                        '.mp4' for video in posVideoList]
        negVideoList = sorted(list(set(videos) - set(posVideoList)))

    print("positive videos ...")
    for idx, video in enumerate(posVideoList):
        print(video, idx)
        if idx >= start_v and idx < end_v:
            feats = extractFeatures(
                feature_extractor, video, fps, modality, datasetName)
            if feats is not None:
                if feats.shape[1] == number_of_frames:
                    print("violence")
                    # mediaevaltest_names.write(basename(video).split(".mp4")[0] + " \n")
                    feature_vector = feats if feature_vector is None else np.concatenate(
                        [feature_vector, feats], axis=0)
                    labels.append(1)

    print("negative videos ...")
    for idx, video in enumerate(negVideoList):
        print(video, idx)
        if idx >= start_v and idx < end_v:
            feats = extractFeatures(
                feature_extractor, video, fps, modality, datasetName)
            if feats is not None:
                if feats.shape[1] == number_of_frames:
                    print("non violence")
                    # mediaevaltest_names.write(basename(video).split(".mp4")[0] + " \n")
                    feature_vector = feats if feature_vector is None else np.concatenate(
                        [feature_vector, feats], axis=0)
                    labels.append(0)

    print("Time spent: {}".format(time.time() - start))
    feature_vector = np.array(feature_vector, dtype=np.float)
    labels = np.array(labels)
    # mediaevaltest_names.close()

    h5.create_dataset('x', data=feature_vector)
    h5.create_dataset('y', data=labels)
    h5.close()


if __name__ == '__main__':
    # start point
    main()
