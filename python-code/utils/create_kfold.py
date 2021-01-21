import numpy as np
import glob
import json
from os.path import basename

dataset_path="/Users/marcostexeira/raw_vf/*"
outfilename="kfold.json"
datapath_features="/Users/marcostexeira/pay-attention-pytorch/data/raw_frames/violentflow/*"

valid_name_videos = [basename(vid) for vid in glob.glob(datapath_features)]

splits = glob.glob(dataset_path)
split_videos = [glob.glob(split+"/vf/*") for split in splits]

dict_data={}

for split_idx, split in enumerate(split_videos):
    vid_list=[]
    for idx, video in enumerate(split):
        vid_name = basename(video)
        if vid_name.endswith("nonviolent"):
            vid_name = vid_name[:-10]
        elif vid_name.endswith("violent"):
            vid_name = vid_name[:-7]
        if not vid_name in valid_name_videos:
            continue

        vid_list.append(vid_name)

    dict_data[split_idx] = vid_list

json_data = json.dumps(dict_data)
with open(outfilename, 'w',encoding='utf-8') as outfile:
    json.dump(dict_data, outfile)
