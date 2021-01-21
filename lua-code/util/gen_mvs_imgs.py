import numpy as np
import glob
import argparse
import os
from cv2 import imwrite,imread,cvtColor,COLOR_BGR2GRAY, resize
import time
import sys
import resnet
import subprocess
import glob

# models
import matplotlib.pyplot as plt

from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model

from os.path import basename, join, exists, splitext, isfile
from joblib import Parallel, delayed
import multiprocessing

AUDIO_EXTENSION=".wav"
VIDEO_EXTENSION=".avi"
VIDEO_EXTENSION2=".mp4"
IMG_EXTENSION=".png"
NUMPY_EXTENSION=".npy"
SEQ_LEN=30

def fuseImgs(frame, rawfrm_path, motion_path):
    flow_dx = frame.split('.')[0] + '_dx.tif'
    flow_dy = frame.split('.')[0] + '_dy.tif'

    #~ flow_dx = frame.replace('.tif','_dx.tif')
    #~ flow_dy = frame.replace('.tif','_dy.tif')
    if exists(rawfrm_path + '/' + frame) and exists(motion_path + '/' + flow_dx) and exists(motion_path + '/' + flow_dy):
        raw_frame = imread(rawfrm_path + '/' + frame)

        raw_frame = cvtColor(raw_frame, COLOR_BGR2GRAY)
        raw_frame = resize(raw_frame, (224,224))

        motion = {}

        motion['dx'] = imread(motion_path + '/' + flow_dx)
        motion['dx'] = resize(motion['dx'], (224,224))
        motion['dy'] = imread(motion_path + '/' + flow_dy)
        motion['dy'] = resize(motion['dy'], (224,224))

        fused = np.zeros((raw_frame.shape[0],raw_frame.shape[1],3),np.float64)

        if len(raw_frame.shape) == 2:   # One channel img
            fused[:,:,0] = raw_frame[:,:]
        else:                           # Three channel img
            fused[:,:,0] = raw_frame[:,:,0]

        fused[:,:,1] = motion['dx'][:,:,0]
        fused[:,:,2] = motion['dy'][:,:,0]

        return fused
    else:
        return False


print("Resnet-101")
base_model = resnet.ResNet101(weights='imagenet')   
feat_type='resnet'
model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
def extract_resnet101_features(model, filename_img):
    img = image.load_img(filename_img, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # shape (1, 2048)
    resnet_features = model.predict(x)[0][0][0]
    print(resnet_features)
    return resnet_features

def extract_frames_and_features(video,args, split_folder):
    global model
    if not isfile(video) or splitext(video)[1] == '.npy':
        return

    cmd = "ffmpeg  -i {} -r 30 {}"
    mkdir_cmd = "mkdir {}_mvs"
    mvs_cmd = "./extract_mvs {} frames.list > {}.mvs"
    delete_cmd = "rm -rf {}"
    cp_cmd = "cp {} {}"
    generate_imgs_cmd = "python generate_mvs_imgs.py -m {} -v {} -o {} -os {}"
    feat_type='motion_vectors_resnet'

    video_name = basename(video).split(".")[0]
    if not exists(join(split_folder,video_name)):
        os.mkdir(join(split_folder,video_name))

    output_feats_np = split_folder+'/'+video_name+feat_type+NUMPY_EXTENSION
    if exists(output_feats_np):
        print("Skipping extraction...")
        subprocess.call(delete_cmd.format(join(split_folder,video_name)),shell=True)
        return
    print("Extracting frames")
    subprocess.call(cmd.format(video,join(split_folder,video_name,video_name+'-%07d'+IMG_EXTENSION)),shell=True)

    if not exists(join(split_folder,video_name)+"_mvs"):
        print("Extract Motion Vector")
        # create folder for motion vectors features
        subprocess.call(mkdir_cmd.format(join(split_folder,video_name)),shell=True)
        # call the motion vector extractor
        subprocess.call(mvs_cmd.format(video,join(split_folder,video_name+'_mvs',video_name)),shell=True)

        print("Extract dx,dy images")
        # generate the dx/dy imgs
        subprocess.call(generate_imgs_cmd.format(join(split_folder,video_name+'_mvs'), video_name,
            join(split_folder,video_name+'_mvs'), '224'),shell=True)

        # # delete
        subprocess.call(delete_cmd.format(join(split_folder,video_name+'_mvs',video_name+'.mvs')),shell=True)
        subprocess.call(delete_cmd.format(join(split_folder,video_name)),shell=True)
        #subprocess.call(delete_cmd.format(join(args.data_folder,video_name+VIDEO_EXTENSION2)),shell=True)

        # extract vgg features
        mvs_folder = join(split_folder,video_name+'_mvs')+'/'
        mvs_imgs   = glob.glob(mvs_folder+'*')
        visual_features=[]

        if len(mvs_imgs)/2 < SEQ_LEN:
            return

        for mvs_img in mvs_imgs:
            if "_dx.tif" in mvs_img:
                continue
            feats = extract_resnet101_features(model,mvs_img)
            visual_features.append(feats)
        visual_features = np.array(visual_features)
        visual_features = visual_features.reshape((visual_features.shape[1],visual_features.shape[0]))

        np.save(output_feats_np,visual_features)
    else:
        visual_features=[]
        valid_count=0
        video_imgs = glob.glob(join(split_folder,video_name)+'/*')
        print("Extracting features ...")
        for img in video_imgs:

            if valid_count > SEQ_LEN:
                break
            elif valid_count > 0:
                fused = fuseImgs(basename(img),join(split_folder,video_name),join(split_folder,video_name)+"_mvs")

                if 'ndarray' in str(type(fused)):
                    fused = np.expand_dims(fused, axis=0)
                    fused = preprocess_input(fused)
                    feats = model.predict(fused)
                    visual_features.append(feats.flatten())
                    valid_count+=1
            else:
                # skipping first frame
                valid_count+=1

        visual_features = np.array(visual_features)
        visual_features = visual_features.reshape((visual_features.shape[1],visual_features.shape[0]))

        np.save(output_feats_np,visual_features)

        subprocess.call(delete_cmd.format(join(split_folder,video_name)),shell=True)




def load_args():
    ap = argparse.ArgumentParser(description='Generates images from Motion Vector information extracted using "extract_mvs". Wiil generate images to ALL of the frames which contains motion information on the file.')

    ap.add_argument('-m', '--mvs-file-path',
                    dest='mvs_path',
                    help='path to the mvs files.',
                    type=str, required=False)
    ap.add_argument('-d', '--dataset_folder',
                    dest='dataset_folder',
                    help='path to the movies dataset.',
                    type=str, default='data/movies_mp4/')
    ap.add_argument('-l', '--videos-list',
                    dest='videos_list_path',
                    help='path to the list of videos.',
                    type=str, required=False)
    ap.add_argument('-o', '--output-path',
                    dest='output_path',
                    help='path to output the extracted frames.',
                    type=str, required=False)
    ap.add_argument('--split_name_folder', dest='split_name_folder',
                    help='folder with videos.',
                    default='vf_motion_vectors')

    args = ap.parse_args()

    return args


def main(args):
    dataset_folder  = args.dataset_folder

    splits = glob.glob(dataset_folder+'*')
    for idx,split in enumerate(splits):
        for video in glob.glob(join(split,args.split_name_folder)+"/*"):
            extract_frames_and_features(video,args,join(split,args.split_name_folder))
    # for split in splits:
    #     print("Split : "+ split)
    #     videos = split+"/"

    #     print("[Dataset Parser] Starting frames extraction")

    #     #extract_frames_from_videos(args,videos,-1,split)
    #     print("Extracting visual features")
    #     extract_features(args,videos)

        #
        # print("cleaning garbage")
        # clean_garbage(train_folder,test_folder)

    print("Done.")

if __name__ == '__main__':
    args = load_args()
    print("....  Starting motion extraction  .....")
    main(args)
# def main():
#     args = load_args()

#     dataset_path  = args.dataset_folder
#     videos = glob.glob(dataset_path+'*')

#     print "Processing a total of",len(videos),"videos."

#     num_cores = multiprocessing.cpu_count()

#     print "Using",num_cores,"cores."
#     for video  in videos:
#         extract_frames(video, args)
#         exit(1)

#     #Parallel(n_jobs=num_cores)(delayed(extract_frames)(video, args) for video in videos)

# if __name__ == '__main__':
#     main()
