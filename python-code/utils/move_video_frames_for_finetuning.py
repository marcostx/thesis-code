import os
from os.path import basename, join, exists, splitext, dirname
import glob
import time
import argparse
import subprocess

VIDEO_EXTENSION = "*.avi"


def getArgs():
    argparser = argparse.ArgumentParser(description=__doc__)

    #argparser.add_argument('-dp','--datapath', default='/Users/marcostexeira/pay-attention-pytorch/data/raw_frames/violentflow', help='Directory containing data sequences', type=str)
    argparser.add_argument('-dp', '--datapath', default='/home/datasets/',
                           help='Directory containing data sequences', type=str)
    argparser.add_argument('-sn', '--split_name',
                           default='train', help='(train/val/test)', type=str)

    args = argparser.parse_args()
    return args


def cp(path, dest):
    cmd_cp = "cp {} {}"
    subprocess.call(cmd_cp.format(path, dest), shell=True)


def main():
    start = time.time()

    args = getArgs()
    datasetPath = args.datapath
    split_name = args.split_name

    positive_folder = "Fight"
    negative_folder = "NonFight"

    neg_folder = join(datasetPath, "rwf-2000-frames-of",
                      split_name, negative_folder)
    pos_folder = join(datasetPath, "rwf-2000-frames-of",
                      split_name, positive_folder)

    posVideoList = glob.glob(
        join(datasetPath, split_name, positive_folder, "*"))
    # non violent videos
    negVideoList = glob.glob(
        join(datasetPath, split_name, negative_folder, "*"))

    for idx, video in enumerate(posVideoList):
        print(video)
        if os.path.isdir(video) and "_flow" in basename(video):
            imgs = glob.glob(join(video, "*.png"))
            for img in imgs:
                print(img)
                cp(img, pos_folder)

    for idx, video in enumerate(negVideoList):
        print(video)
        if os.path.isdir(video) and "_flow" in basename(video):
            imgs = glob.glob(join(video, "*.png"))
            for img in imgs:
                print(img)
                cp(img, neg_folder)

    end = time.time()
    print("Time spent: {}".format(end - start))


if __name__ == '__main__':
    main()
