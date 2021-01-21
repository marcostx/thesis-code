import numpy as np
import glob
import argparse
import os
import cv2
import resnet
import sys
import scipy.io.wavfile as wav
import subprocess

# models
from keras.applications.vgg19 import VGG19
from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model

from os.path import basename, join, exists, splitext
from sklearn.model_selection import train_test_split
from skimage.feature import hog

AUDIO_EXTENSION=".wav"
VIDEO_EXTENSION=".avi"
VIDEO_EXTENSION2=".mpg"
IMG_EXTENSION=".png"
NUMPY_EXTENSION=".npy"
SEQ_LEN=30
SIFT_LEN_USED=100
VISUAL_SIZE=4096
MOTION_SIZE=3072



def clean_garbage(train_folder, test_folder):
    cmd = 'rm -rf {}*{}'
    subprocess.call(cmd.format(train_folder,IMG_EXTENSION),shell=True)
    subprocess.call(cmd.format(test_folder,IMG_EXTENSION),shell=True)


def extract_resnet101_features(model, filename_img):
    img = image.load_img(filename_img, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # shape (1, 2048)
    resnet_features = model.predict(x)[0][0][0]

    return resnet_features

def extract_inceptionresnet_features(model, filename_img):
    img = image.load_img(filename_img, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # shape (1, 1536)
    net_feats = model.predict(x)

    return net_feats[0]


def extract_vgg_features(model, filename_img):
    img = image.load_img(filename_img, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # shape (1, 4096)
    vgg_features = model.predict(x)

    return vgg_features[0]


# hog descriptor
def extract_HOG(img, orient=8, pixelPerCell=12, cellPerBlock=1,
                    vis=False, featureVec=True):
    features = hog(img, orientations=orient,
                                  pixels_per_cell=(pixelPerCell, pixelPerCell),
                                  cells_per_block=(cellPerBlock, cellPerBlock),
                                  transform_sqrt=True,
                                  visualise=vis, block_norm='L2-Hys',feature_vector=featureVec)
    return features[:MOTION_SIZE]

def extract_frames_from_videos(data, labels, split,sampling=128000):
    # videos = glob.glob(dataset_video_folder+'*'+VIDEO_EXTENSION)

    if not exists(split):
        os.mkdir(split)

    cmd = "ffmpeg  -i {} -r 25 {}"

    print("[Dataset Frames Parser] Number of videos : " , len(data))
    # extract audio for each video
    for idx,video in enumerate(data):
        print(video)
        video_name = basename(video).split(".")[0]
        video_frames_folder = video.split(".")[0]

        class_=None
        if labels[idx] == 1:
            class_ = 'violent'
        else:
            class_ = 'nonviolent'

        if exists(join(split,video_name+class_)):
            print("Skipping video ...")
            continue

        if not exists(join(split,video_name+class_)):
            os.mkdir(join(split,video_name+class_))

        subprocess.call(cmd.format(video,join(split,video_name+class_,video_name+'%03d'+IMG_EXTENSION)),shell=True)



def extract_features(args,split):

    video_folders = os.listdir(split)

    if args.visual_model == 'vgg':
        print("VGG19")
        # initialize vgg model
        base_model = VGG19(weights='imagenet')
        feat_type = "_vgg"
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
    elif args.visual_model == 'incept-rest-v2':
        # initialize inception resnet v2 model
        print("Inception-Resnet-V2")
        base_model = InceptionResNetV2(weights='imagenet')
        feat_type='_inc-resm-v2'
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
    elif args.visual_model == 'resnet':
        # initialize inception resnet v2 model
        print("Resnet-101")
        base_model = resnet.ResNet101(weights='imagenet')
        feat_type='resnet'
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)



    for idx,video in enumerate(video_folders):
        if not os.path.isdir(split+video):
            continue

        print(" Extraindo features : ", split+video)
        output_feats_np = split+video+feat_type+NUMPY_EXTENSION
        if exists(output_feats_np):
            print("Skipping extraction...")
            continue

        video_frames = glob.glob(split+video+'/*'+IMG_EXTENSION)

        if len(video_frames) < 30:
            continue

        visual_features = []
        for i in range(int(len(video_frames)/2) - int(SEQ_LEN/2),int(len(video_frames)/2) + int(SEQ_LEN/2)):
            #img_   = cv2.imread(video_frame[i])
            if args.visual_model == 'vgg':
                feats = extract_vgg_features(model,video_frames[i])
            elif args.visual_model == 'incept-rest-v2':
                feats = extract_inceptionresnet_features(model, video_frames[i])
            elif args.visual_model == 'resnet':
                feats = extract_resnet101_features(model, video_frames[i])

            visual_features.append(feats)

        visual_features = np.array(visual_features)
        visual_features = visual_features.reshape((visual_features.shape[1],visual_features.shape[0]))
        print("Salvando npy arquivo!!")
        np.save(output_feats_np,visual_features)


def extract_hog_features(split):
    video_folders = os.listdir(split)

    for idx,video in enumerate(video_folders):
        if not os.path.isdir(split+video):
            continue

        print(" Extraindo HOG features : ", split+video)
        output_feats_np = split+video+'_hog'+NUMPY_EXTENSION
        video_frames = glob.glob(split+video+'/*'+IMG_EXTENSION)
        if len(video_frames) == 0:
            continue
        # if exists(output_feats_np):
        #     print("Skipping extraction...")
        #     continue

        hog_features = []
        if len(video_frames) < 30:
            continue

        # # pegando o meio da sequencia
        for i in range(int(len(video_frames)/2) - int(SEQ_LEN/2),int(len(video_frames)/2) + int(SEQ_LEN/2)):
            img_   = cv2.imread(video_frames[i])
            gray_img = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
            img_hog = extract_HOG(gray_img)

            hog_features.append(img_hog)

        hog_features = np.array(hog_features)
        hog_features = hog_features.reshape((hog_features.shape[1],hog_features.shape[0]))

        # np_del 'violent_hog'+NUMPY_EXTENSION
        # cmd = 'rm -rf *{}'
        # print("deletando numpys incorretos")
        # subprocess.call(cmd.format(output_feats_np_del),shell=True)

        np.save(output_feats_np,hog_features)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Extracting audio feature from videos.')
    parser.add_argument('--violence_folder', dest='violence_folder', help='folder with violent videos.', default='Violence')
    parser.add_argument('--non_violence_folder', dest='non_violence_folder', help='folder with non violent videos.', default='NonViolence')
    parser.add_argument('--dataset_splits', dest='dataset_splits', help='folder with spits of cross validation.', default='data/movies_seqlen30/')
    parser.add_argument('--if_movies', dest='if_movies', help='dataset selection (vf/movies)', default=True)
    parser.add_argument('--if_hog', dest='if_hog', help='use hog descriptor or not', default=1)
    parser.add_argument('--visual_model', dest='visual_model', help='pretrained net used for feature extraction', default='resnet')
    parser.add_argument('--movies_folder', dest='movies_folder', help='name of folder containing Movies videos', default="movies")

    args = parser.parse_args()

    return args


def parse_dataset(args):
    violence_dataset_path  = args.violence_folder
    non_violence_dataset_path  = args.non_violence_folder
    dataset_splits  = args.dataset_splits

    splits = glob.glob(dataset_splits+'*')

    for split in splits:
        print("Split : "+ split)

        violent_path_name=join(split, violence_dataset_path)
        non_violent_path_name=join(split, non_violence_dataset_path)

        # dataset split
        violent_videos = glob.glob(violent_path_name+'/*'+VIDEO_EXTENSION)
        non_violent_videos = glob.glob(non_violent_path_name+'/*'+VIDEO_EXTENSION)
        train_folder = split+'/train/'
        test_folder = split+'/test/'

        if not exists(train_folder):
            os.mkdir(train_folder)
        if not exists(test_folder):
            os.mkdir(test_folder)

        data = violent_videos + non_violent_videos
        labels = list([1] * len(violent_videos)) + list([0] * len(non_violent_videos))
        X_train, X_test, y_train, y_test = train_test_split(data,labels, test_size=0.2,random_state=42)

        print("[Dataset Parser] Starting frames extraction")

        # extract_frames_from_videos(X_train,y_train,train_folder)
        # extract_frames_from_videos(X_test,y_test,test_folder)

        # print("Extracting visual features")
        extract_features(args,train_folder)
        extract_features(args,test_folder)

        if args.if_hog == 1:
            print("Extracting HOG")
            extract_hog_features(train_folder)
            extract_hog_features(test_folder)
        print("Saindo!")


        #
        # print("cleaning garbage")
        # clean_garbage(train_folder,test_folder)

    print("Done.")


# similiar to parse_dataset() but for Movies dataset
# which is located inside violentflows folder splits
def parse_movies_dataset(args):

    violence_dataset_path  = args.violence_folder
    non_violence_dataset_path  = args.non_violence_folder
    dataset_splits  = args.dataset_splits
    movies_folder  = args.movies_folder


    splits = glob.glob(dataset_splits+'*')
    for split in splits:
        print("Split : "+ split)

        # e.g. data/movies/1/Violence/movies/
        violent_path_name=join(split, violence_dataset_path, movies_folder)
        non_violent_path_name=join(split, non_violence_dataset_path, movies_folder)

        # dataset split
        violent_videos = glob.glob(violent_path_name+'/*'+VIDEO_EXTENSION) + glob.glob(violent_path_name+'/*'+VIDEO_EXTENSION2)
        non_violent_videos = glob.glob(non_violent_path_name+'/*'+VIDEO_EXTENSION) + glob.glob(non_violent_path_name+'/*'+VIDEO_EXTENSION2)
        movies_folder = split+'/movies/'

        data = violent_videos + non_violent_videos

        labels = list([1] * len(violent_videos)) + list([0] * len(non_violent_videos))
        X_train, X_test, y_train, y_test = train_test_split(data,labels, test_size=0.2,random_state=42)

        print("[Dataset Parser] Starting frames extraction")

        #extract_frames_from_videos(data,labels,movies_folder)

        # print("Extracting visual features")
        #extract_features(args,movies_folder)

        if args.if_hog == 1:
            print("Extracting HOG")
            extract_hog_features(movies_folder)

        print("Saindo!")


if __name__ == '__main__':
    args = parse_args()
    if args.if_movies:
        print("....  Movies Dataset  .....")
        parse_movies_dataset(args)
    else:
        print("....  ViolentFlows Dataset  .....")
        parse_dataset(args)
