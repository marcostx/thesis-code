import numpy as np
import glob
import argparse
import os
import cv2
import time
import sys
import subprocess
import glob

# models
import matplotlib.pyplot as plt

from os.path import basename, join, exists, splitext
from joblib import Parallel, delayed  
import multiprocessing

AUDIO_EXTENSION=".wav"
VIDEO_EXTENSION=".avi"
VIDEO_EXTENSION2=".mp4"
IMG_EXTENSION=".png"
NUMPY_EXTENSION=".npy"
SEQ_LEN=100


def extract_frames(video,args):
    if not exists(args.data_folder):
        os.mkdir(args.data_folder)
    cmd = "ffmpeg  -i {} -r 15 {}"
    mkdir_cmd = "mkdir {}_mvs"
    mvs_cmd = "./extract_mvs {} frames.list > {}.mvs"
    delete_cmd = "rm -rf {}"
    cp_cmd = "cp {} {}" 
    generate_imgs_cmd = "python generate_mvs_imgs.py -m {} -v {} -o {} -os {}"

    video_name = basename(video).split(".")[0]

    # moving to where theres space
    subprocess.call(cp_cmd.format(video,args.data_folder), shell=True)
    cp_video = join(args.data_folder,video_name+VIDEO_EXTENSION2)
    if not exists(join(args.data_folder,video_name)):
        os.mkdir(join(args.data_folder,video_name))

    print("Extracting frames")
    subprocess.call(cmd.format(cp_video,join(args.data_folder,video_name,video_name+'%03d'+IMG_EXTENSION)),shell=True)

    print("Extract Motion Vector")
    # create folder for motion vectors features
    subprocess.call(mkdir_cmd.format(join(args.data_folder,video_name)),shell=True)
    # call the motion vector extractor
    subprocess.call(mvs_cmd.format(cp_video,join(args.data_folder,video_name+'_mvs',video_name)),shell=True)

    print("Extract dx,dy images")
    # generate the dx/dy imgs
    subprocess.call(generate_imgs_cmd.format(join(args.data_folder,video_name+'_mvs'), video_name, 
        join(args.data_folder,video_name+'_mvs'), '244'),shell=True)

    # # delete
    #subprocess.call(delete_cmd.format(join(args.data_folder,video_name+'_mvs',video_name+'.mvs')),shell=True)
    #subprocess.call(delete_cmd.format(join(args.data_folder,video_name)),shell=True)
    subprocess.call(delete_cmd.format(join(args.data_folder,video_name+VIDEO_EXTENSION2)),shell=True)

    print("Done")




def load_args():
    ap = argparse.ArgumentParser(description='Generates images from Motion Vector information extracted using "extract_mvs". Wiil generate images to ALL of the frames which contains motion information on the file.')

    ap.add_argument('-m', '--mvs-file-path',
                    dest='mvs_path',
                    help='path to the mvs files.',
                    type=str, required=False)
    ap.add_argument('-d', '--dataset_folder',
                    dest='dataset_folder',
                    help='path to the movies dataset.',
                    type=str, default='dataset/train/')
    ap.add_argument('-t', '--data_folder',
                    dest='data_folder',
                    help='',
                    type=str, default='data/train_motion_vectors/')
    ap.add_argument('-l', '--videos-list',
                    dest='videos_list_path',
                    help='path to the list of videos.',
                    type=str, required=False)
    ap.add_argument('-o', '--output-path',
                    dest='output_path',
                    help='path to output the extracted frames.',
                    type=str, required=False)

    args = ap.parse_args()
    
    return args

def main():
    args = load_args()

    dataset_path  = args.dataset_folder
    videos = glob.glob(dataset_path+'*')

    print "Processing a total of",len(videos),"videos."
    
    num_cores = multiprocessing.cpu_count()

    print "Using",num_cores,"cores."
    for video  in videos:
        extract_frames(video, args)
        exit(1)
    
    #Parallel(n_jobs=num_cores)(delayed(extract_frames)(video, args) for video in videos)

if __name__ == '__main__':
    main()