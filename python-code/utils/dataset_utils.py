import os
import json
import glob
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import torch

def make_split(data_dir):
    dataset = []
    labels = []
    for target in sorted(os.listdir(data_dir)):
        d = os.path.join(data_dir, target)
        if not os.path.isdir(d):
            continue
        if "violence" in d:
            labels.append(1)
        else:
            labels.append(0)
        dataset.append(d)

    train_path, test_path, train_y, test_y =  train_test_split(dataset,labels, test_size=0.20, stratify=labels)
    return train_path, train_y, test_path, test_y

def early_fusion(f1, f2):

    merged_x = np.concatenate((f1['x'], f2['x']), axis=2)

    return torch.Tensor(merged_x)

def get_images(trainx, trainy, valx, valy):
    n_trainx=[]
    n_trainy=[]
    n_valx=[]
    n_valy=[]

    for idx, vid_train in enumerate(trainx):
        for frame in glob.glob(os.path.join(vid_train,"*.png")):
            n_trainx.append(frame)
            n_trainy.append(trainy[idx])
    for idx, vid_val in enumerate(valx):
        for frame in glob.glob(os.path.join(vid_val,"*.png")):
            n_valx.append(frame)
            n_valy.append(valy[idx])
            

    return n_trainx, n_trainy, n_valx, n_valy

def get_splits(datasetPath):
    train_path = os.path.join(datasetPath, "train")
    val_path = os.path.join(datasetPath, "val")

    positiveFolder="Fight"
    negativeFolder="NonFight"
    VIDEO_EXTENSION="*.avi"
    video_list=[]
    labels=[]

    for ix, split in enumerate([train_path, val_path]):
        # violent videos
        posVideoList = glob.glob(os.path.join(split, positiveFolder, "*"))
        print("positive videos ...")
        for idx,video in enumerate(posVideoList):
            if os.path.isdir(video) and len(glob.glob(os.path.join(video,"*.png")))==150:
                video_list.append(video)
                labels.append(1)

        # non violent videos
        negVideoList = glob.glob(os.path.join(split, negativeFolder, "*"))

        print("negative videos ...")
        for idx,video in enumerate(negVideoList):
            if os.path.isdir(video) and len(glob.glob(os.path.join(video,"*.png")))==150:
                video_list.append(video)
                labels.append(0)

        # train
        if ix==0:
            trainX = video_list
            trainY = labels
        # validation
        if ix==1:
            validX = video_list
            validY = labels

        video_list=[]
        labels=[]

    return trainX, validX, trainY, validY

def load_split_data(train_xs, train_splits, test_xs, test_splits, datapath):
    train_path, train_y, test_path, test_y = [],[],[],[]

    flatten = lambda l: [item for sublist in l for item in sublist]

    for idx, splt in enumerate(train_xs):
        #splt_data = [datapath+"/"+str(train_splits[idx])+"/"+item for item in splt]
        splt_data = [datapath+"/"+item for item in splt]
        train_path.append(splt_data)
        for item in splt:
            if "violence" in item:
                train_y.append(1)
            else:
                train_y.append(0)

    for idx, splt in enumerate(test_xs):
        splt_data = [datapath+"/"+item for item in splt]
        test_path.append(splt_data)
        for item in splt:
            if "violence" in item:
                test_y.append(1)
            else:
                test_y.append(0)

    train_path = flatten(train_path)
    test_path  = flatten(test_path)
    return train_path, train_y, test_path, test_y

def load_kfold_generator(kfold_file_json):

    with open(kfold_file_json, 'r') as kfile:
        kfile = json.load(kfile)
    splits = list(kfile.keys())

    kf = KFold(n_splits=5)
    indices=[]
    for train_index,test_index in kf.split(splits):
        indices.append([train_index,test_index])
    indices=reversed(indices)


    for train_index,test_index in indices:
        print(train_index, test_index)
        X_train = [kfile[str(index)] for index in train_index]
        X_test = [kfile[str(index)] for index in test_index]

        yield X_train, train_index, X_test, test_index

def get_dataset_params(dataset_name, cross=False):
    if dataset_name == "CK":
        if cross:
            return {
                "dataset_name": dataset_name,
                "type_load": "last",
                "n_labels": 6,
                "k": 5,
                "classes": [
                    "Anger",
                    "Disgust",
                    "Fear",
                    "Happiness",
                    "Sadness",
                    "Surprise"
                ]
            }
        else:
            return {
                "dataset_name": dataset_name,
                "type_load": "last",
                "n_labels": 7,
                "k": 5,
                "classes": [
                    "Anger",
                    "Contempt",
                    "Disgust",
                    "Fear",
                    "Happiness",
                    "Sadness",
                    "Surprise"
                ]
            }
    elif dataset_name == "violentflows":
        return {
            "dataset_name": dataset_name,
            "type_load": "-",
            "n_classes": 2,
            "k": 5,
            "classes": [
                "Violence",
                "NonViolence"
            ]
        }
    elif dataset_name == "hockey":
        return {
            "dataset_name": dataset_name,
            "type_load": "-",
            "n_classes": 2,
            "k": 5,
            "classes": [
                "Violence",
                "NonViolence"
            ]
        }

    elif dataset_name == "UCF-101":
        return {
            "dataset_name": dataset_name,
            "type_load": "-",
            "n_labels": 101,
            "classes": [
                "ApplyEyeMakeup",
                "ApplyLipstick",
                "Archery",
                "BabyCrawling",
                "BalanceBeam",
                "BandMarching",
                "BaseballPitch",
                "Basketball",
                "BasketballDunk",
                "BenchPress",
                "Biking",
                "Billiards",
                "BlowDryHair",
                "BlowingCandles",
                "BodyWeightSquats",
                "Bowling",
                "BoxingPunchingBag",
                "BoxingSpeedBag",
                "BreastStroke",
                "BrushingTeeth",
                "CleanAndJerk",
                "CliffDiving",
                "CricketBowling",
                "CricketShot",
                "CuttingInKitchen",
                "Diving",
                "Drumming",
                "Fencing",
                "FieldHockeyPenalty",
                "FloorGymnastics",
                "FrisbeeCatch",
                "FrontCrawl",
                "GolfSwing",
                "Haircut",
                "Hammering",
                "HammerThrow",
                "HandstandPushups",
                "HandstandWalking",
                "HeadMassage",
                "HighJump",
                "HorseRace",
                "HorseRiding",
                "HulaHoop",
                "IceDancing",
                "JavelinThrow",
                "JugglingBalls",
                "JumpingJack",
                "JumpRope",
                "Kayaking",
                "Knitting",
                "LongJump",
                "Lunges",
                "MilitaryParade",
                "Mixing",
                "MoppingFloor",
                "Nunchucks",
                "ParallelBars",
                "PizzaTossing",
                "PlayingCello",
                "PlayingDaf",
                "PlayingDhol",
                "PlayingFlute",
                "PlayingGuitar",
                "PlayingPiano",
                "PlayingSitar",
                "PlayingTabla",
                "PlayingViolin",
                "PoleVault",
                "PommelHorse",
                "PullUps",
                "Punch",
                "PushUps",
                "Rafting",
                "RockClimbingIndoor",
                "RopeClimbing",
                "Rowing",
                "SalsaSpin",
                "ShavingBeard",
                "Shotput",
                "SkateBoarding",
                "Skiing",
                "Skijet",
                "SkyDiving",
                "SoccerJuggling",
                "SoccerPenalty",
                "StillRings",
                "SumoWrestling",
                "Surfing",
                "Swing",
                "TableTennisShot",
                "TaiChi",
                "TennisSwing",
                "ThrowDiscus",
                "TrampolineJumping",
                "Typing",
                "UnevenBars",
                "VolleyballSpiking",
                "WalkingWithDog",
                "WallPushups",
                "WritingOnBoard",
                "YoYo"
            ]
        }
