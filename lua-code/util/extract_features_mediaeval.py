import numpy as np
import glob
import argparse
import os
#import resnet
import cv2
import time
#import librosa
import sys
import subprocess
from scipy import signal
#import librosa.display as dp
import time
from joblib import Parallel, delayed
import multiprocessing

# models
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
#from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
import matplotlib.pyplot as plt
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences

from os.path import basename, join, exists, splitext
from sklearn.model_selection import train_test_split
from skimage.feature import hog

AUDIO_EXTENSION=".wav"
VIDEO_EXTENSION=".avi"
VIDEO_EXTENSION2=".mp4"
IMG_EXTENSION=".png"
NUMPY_EXTENSION=".npy"
SEQ_LEN=100
VISUAL_SIZE=4096
MOTION_SIZE=1200





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

def extract_inception_features(model, filename_img):
    img = image.load_img(filename_img, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # shape (1, 2048)
    inception_features = model.predict(x)[0][0]

    return inception_features[0]

def extract_vgg_features(model, filename_img):
    img = image.load_img(filename_img, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # shape (1, 4096)
    vgg_features = model.predict(x)

    return vgg_features[0]


# hog descriptor
def extract_HOG(img, orient=8, pixelPerCell=30, cellPerBlock=1,
                    vis=False, featureVec=True):
    features = hog(img, orientations=orient,
                                  pixels_per_cell=(pixelPerCell, pixelPerCell),
                                  cells_per_block=(cellPerBlock, cellPerBlock),
                                  transform_sqrt=True,
                                  visualise=vis, block_norm='L2-Hys',feature_vector=featureVec)
    return features[:MOTION_SIZE]

# mfcc
def mfcc(audio, sample_rate, hop_length, n_fft, n_mfcc=40):
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)

    return mfccs

def train_violence_parser(file_inp,dataset_path):
    lines = file_inp.readlines()
    lines = [str.replace("\n","") for str in lines]
    video_files = glob.glob(dataset_path+'*.mp4')
    video_files_wt_ext = [basename(video).split(".")[0] for video in video_files]

    y = list(filter(lambda video: video in lines, video_files_wt_ext))

    return y

def test_violence_parser(file_inp,dataset_path):
    lines = file_inp.readlines()
    lines = [str.replace("\n","") for str in lines]
    violence = []
    for line in lines:

        if line.split(" ")[3] == "1":
            violence.append(line.split(" ")[2])
    #lines = [str.replace("\n","") for str in lines]
    video_files = glob.glob(dataset_path+'*.mp4')

    video_files_wt_ext = [basename(video).split(".")[0] for video in video_files]

    y = list(filter(lambda video: video in violence, video_files_wt_ext))

    return y

def extract_audio_and_features(args, data, dataset_path):
    len_mfcc = []

    base_model = VGG19(weights='imagenet')

    model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)

    # nome dos videos de violencia
    if args.istrain:
        args.violece_file = 'violence.txt'
        vl = open(args.violece_file)
        targets_fn = train_violence_parser(vl,args.dataset_folder)
    else:
        args.violece_file = 'violence_test.txt'
        vl = open(args.violece_file)
        targets_fn = test_violence_parser(vl,args.dataset_folder)

    cmd = "ffmpeg  -i {} -f wav -vn {} -y"
    cmd_split = "ffmpeg -i {} -f segment -segment_time 0.06 -c copy {}"
    cmd_delete = "rm -rf {}"
    cmd_mov = "cp {} {}"

    print("[Dataset Frames Parser] Number of videos : " , len(data))
    start = time.time()
    # extract audio for each video
    for idx,video in enumerate(data):
        video_name = basename(video).split(".")[0]
        print(video_name)
        ext = splitext(basename(video))[1]
        # skipping .npy files
        if (not ext == VIDEO_EXTENSION2):
            continue

        if video_name in targets_fn:
            output_feats_np = join(args.npy_folder,video_name)+"spec"+"violence"+NUMPY_EXTENSION
        elif video_name not in targets_fn:
            output_feats_np = join(args.npy_folder,video_name)+"spec"+"nonviolence"+NUMPY_EXTENSION

        if exists(output_feats_np):
            print("Skipping extraction...")
            continue

        # moving to where theres space
        subprocess.call(cmd_mov.format(video,args.npy_folder), shell=True)
        video = join(args.npy_folder,video_name+VIDEO_EXTENSION2)
        if not exists(join(args.npy_folder,video_name)):
            os.mkdir(join(args.npy_folder,video_name))

        print("Extracting audio")
        subprocess.call(cmd.format(video,join(args.npy_folder,video_name,video_name+AUDIO_EXTENSION)),shell=True)

        # removendo video
        subprocess.call(cmd_delete.format(video), shell=True)

        print("Splitting audio")
        # split audio
        subprocess.call(cmd_split.format(join(args.npy_folder,video_name,video_name+AUDIO_EXTENSION),
                        join(args.npy_folder,video_name,video_name+'%03d'+AUDIO_EXTENSION)),shell=True)
        # removing original audio
        subprocess.call(cmd_delete.format(join(args.npy_folder,video_name,video_name+AUDIO_EXTENSION)), shell=True)

        audio_segments = glob.glob(join(args.npy_folder,video_name)+'/*'+AUDIO_EXTENSION)

        # skip short clips
        if len(audio_segments) < SEQ_LEN:
            continue

        spec_features = []
        print("Extracting features")
        # middle of the video
        for i in range(int((len(audio_segments)/2)-(SEQ_LEN)/2)-1,int(((len(audio_segments)/2)+(SEQ_LEN)/2))-1):
            # reading file
            audio_file = audio_segments[i]

            y, sr = librosa.load(audio_file, sr=None)

            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256)
            mel_db = librosa.power_to_db(S, ref=np.max)
            fig1, ax = plt.subplots()
            spec = dp.specshow(mel_db,y_axis='mel', fmax=8000,
                             x_axis='time',ax=ax)
            fig1.canvas.draw()
            buf = fig1.canvas.tostring_rgb()
            buf = np.fromstring(buf, dtype=np.uint8)
            ncols, nrows = fig1.canvas.get_width_height()
            img = np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)

            img_res = img[55:img.shape[0]-55,80:img.shape[1]-80,:]

            img_res = cv2.resize(img_res, (224,224))

            x = image.img_to_array(img_res)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            #cv2.imwrite('teste2.png',img_res)
            feats = model.predict(x)[0]
            spec_features.append(feats)

        spec_features = np.array(spec_features)
        spec_features = spec_features.reshape((spec_features.shape[1],spec_features.shape[0]))

        print("Salvando npy arquivo!!")
        np.save(output_feats_np,spec_features)

        print("Removendo frames")
        subprocess.call(cmd_delete.format(join(args.npy_folder, video_name)), shell=True)

        # # calc mfccs
        # feats = mfcc(y,sr,int(0.01*sr), n_fft=int(0.025*sr))

        # # padding
        # feats = pad_sequences(feats, maxlen=SEQ_LEN, padding='post', truncating='post')

        # # skip short clips
        # # if feats.shape[1] < SEQ_LEN:
        # #     continue

        # feats = feats[:SEQ_LEN]
        # len_mfcc.append(feats.shape[1])
        # print("Salvando npy arquivo!!")
        # np.save(output_feats_np,feats)

        # #print("Removendo frames")
        # subprocess.call(cmd_delete.format(join(args.npy_folder, video_name)), shell=True)








# augment positive file with transformations (transpose, flip)
def augment_positive_videos(args, videos, dataset_path):
    vl = open(args.violece_file)
    lines = vl.readlines()
    lines = [str.replace("\n","") for str in lines]
    video_files = glob.glob(dataset_path+'*.mp4')
    video_files_wt_ext = [video.split(".")[0] for video in video_files]

    # nome dos videos de violencia
    targets_fn = list(filter(lambda video: basename(video) in lines, video_files_wt_ext))

    # numpys para renomear
    np_files = glob.glob(dataset_path+'*.mp4')

    # command
    cmd = "ffmpeg -i {} -vf {} -c:a copy {}"

    for idx, video_file in enumerate(np_files):

        if splitext(video_file)[0] in targets_fn:
            original = video_files
            print("Video:  {}, file : {}".format(original, idx))

            print("Horizontal Flipping")
            trans_type = 'hflip'
            output_transformed  = video_file.split(".")[0] + trans_type + splitext(video_file)[1]
            subprocess.call(cmd.format(original,args.transformation ,output_transformed), shell=True)
            print("Vertical Flipping")
            trans_type = 'vflip'
            output_transformed  = video_file.split(".")[0] + trans_type + splitext(video_file)[1]
            subprocess.call(cmd.format(original,args.transformation,output_transformed), shell=True)
            print("Transposing")
            trans_type = 'transpose=1'
            output_transformed  = video_file.split(".")[0] + 'transpose' + splitext(video_file)[1]
            subprocess.call(cmd.format(original,trans_type ,output_transformed), shell=True)
        else:
            print("Skipping video")

    print("Done!")

def extract_frames_and_features(video, args):

    print("Inception V3")
    base_model = InceptionV3(weights='imagenet')

    model = Model(input=base_model.input, output=base_model.get_layer('avg_pool').output)

    if args.istrain:
        args.violece_file = 'violence.txt'
        vl = open(args.violece_file)
        targets_fn = train_violence_parser(vl,args.dataset_folder)
    else:
        args.violece_file = 'violence_test.txt'
        vl = open(args.violece_file)
        targets_fn = test_violence_parser(vl,args.dataset_folder)

    cmd = "ffmpeg  -i {} -r 15 {}"
    cmd_delete = "rm -rf {}"
    cmd_mov = "cp {} {}"

    video_name = basename(video).split(".")[0]
    ext = splitext(basename(video))[1]
    # skipping .npy files
    if (not ext == VIDEO_EXTENSION2):
        return

    if video_name in targets_fn:
        output_feats_np = join(args.npy_folder,video_name)+args.feature_extractor+"violence"+NUMPY_EXTENSION
    elif video_name not in targets_fn:
        output_feats_np = join(args.npy_folder,video_name)+args.feature_extractor+"nonviolence"+NUMPY_EXTENSION

    if exists(output_feats_np):
        print("Skipping extraction...")
        return
    # restringindo apenas para transformacoes


    # moving to where theres space
    subprocess.call(cmd_mov.format(video,args.npy_folder), shell=True)
    video = join(args.npy_folder,video_name+VIDEO_EXTENSION2)
    if not exists(join(args.npy_folder,video_name)):
        os.mkdir(join(args.npy_folder,video_name))

    print("Extracting frames")
    subprocess.call(cmd.format(video,join(args.npy_folder,video_name,video_name+'%03d'+IMG_EXTENSION)),shell=True)

    #removendo video
    subprocess.call(cmd_delete.format(video), shell=True)
    # extracting features
    video_frames = glob.glob(join(args.npy_folder,video_name)+'/*')

    # skip short clips
    if len(video_frames) < SEQ_LEN:
        return

    visual_features = []
    print("Extracting features")
    # middle of the video
    for i in range(int((len(video_frames)/2)-(SEQ_LEN)/2),int(((len(video_frames)/2)+(SEQ_LEN)/2))-1):
        #img_   = cv2.imread(video_frame[i])
        if args.feature_extractor == 'vgg':
            feats = extract_vgg_features(model,video_frames[i])
        elif args.feature_extractor == 'resnet':
            feats = extract_resnet101_features(model, video_frames[i])
        elif args.feature_extractor == 'incept':
            feats = extract_inception_features(model, video_frames[i])
        elif args.feature_extractor == 'hog':
            img_   = cv2.imread(video_frames[i])
            gray_img = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
            gray_img = cv2.resize(gray_img,(480,320))
            feats = extract_HOG(gray_img)

        visual_features.append(feats)

    visual_features = np.array(visual_features)
    visual_features = visual_features.reshape((visual_features.shape[1],visual_features.shape[0]))

    end = time.time() - start
    print("tempo gasto : ", end)
    print("Salvando npy arquivo!!")
    np.save(output_feats_np,visual_features)

    print("Removendo frames")
    subprocess.call(cmd_delete.format(join(args.npy_folder, video_name)), shell=True)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Extracting audio feature from videos.')
    parser.add_argument('--dataset_folder', dest='dataset_folder', help='folder with videos.', default='../mediaeval2015_subset/test/')
    parser.add_argument('--npy_folder', dest='npy_folder', help='folder with numpy files.', default='data/train_incept/')
    parser.add_argument('--feature_extractor', dest='feature_extractor', help='', default='incept')
    parser.add_argument('--augmentation', dest='aug', help='', default=False)
    parser.add_argument('--transformation', dest='transformation', help='', default=None)
    parser.add_argument('--violece_file', dest='violece_file', help='', default='')
    parser.add_argument('--istrain', dest='istrain', help='', default=True)
    parser.add_argument('--isaudio', dest='isaudio', help='', default=False)

    args = parser.parse_args()

    return args


def parse_dataset():
    args = parse_args()
    dataset_path  = args.dataset_folder
    
    videos = glob.glob(dataset_path+'*')
    

    # # nome dos videos de violencia

    # if args.feature_extractor == 'vgg':
    #     print("VGG19")
    #     # initialize vgg model
    #     base_model = VGG19(weights='imagenet')

    #     model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)
    # elif args.feature_extractor == 'resnet':
    #     # initialize resnet-101 model
    #     print("Resnet-101")
    #     base_model = resnet.ResNet101(weights='imagenet')

    #     model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
    # elif args.feature_extractor == 'incept':
    #     # initialize resnet-101 model
    #     print("Inception V3")
    #     base_model = InceptionV3(weights='imagenet')

    #     model = Model(input=base_model.input, output=base_model.get_layer('avg_pool').output)
    # elif args.feature_extractor == 'hog':
    #     # no model dependency
    #     print("HOG")

    print("Processing a total of {} videos.".format(len(videos)))

    num_cores = multiprocessing.cpu_count()

    print("Using {} cores.".format(num_cores))
    print("[Dataset Parser] Starting frames extraction")

    Parallel(n_jobs=num_cores)(delayed(extract_frames_and_features)(video,args) for video in videos)

    # if not args.isaudio:
    #     if args.aug == True:
    #         print("[Data Agumentation] Performing data augmentation in positive class")
    #         augment_positive_videos(args, videos, dataset_path)
    #     else:
    #         print("[Dataset Parser] Starting frames extraction")
    #         extract_frames_and_features(args,videos,dataset_path)
    # elif args.isaudio:
    #     print("[Dataset Parser] Starting audio extraction")
    #     extract_audio_and_features(args, videos, dataset_path)



    print("Done.")

# start point
parse_dataset()
