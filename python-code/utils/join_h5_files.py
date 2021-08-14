import h5py
import numpy as np
import os
import subprocess
import glob

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

files = [
    'mediaeval-test-0-500.h5',
    'mediaeval-test-500-1000.h5',
    'mediaeval-test-1000-1500.h5',
    'mediaeval-test-1500-2000.h5',
    'mediaeval-test-2000-2500.h5',
    'mediaeval-test-2500-3000.h5',
    'mediaeval-test-3000-3500.h5',
    'mediaeval-test-3500-4000.h5',
    'mediaeval-test-4000-4520.h5']

output_file = 'mediaeval-test-efficientnet-flow.h5'
h5 = h5py.File(output_file, 'w')

params = {}
side_info = {}
attrs = {}

for file in files:

    with h5py.File(file, 'r') as f:
        for k, v in f.attrs.items():
            attrs[k] = v
        for p in f.keys():
            if p in params:
                print(file)
                print(f[p].shape)
                params[p] = np.append(params[p], np.array(
                    f[p]).astype('float32'), axis=0)
            else:
                print(f[p].shape)
                params[p] = np.array(f[p]).astype('float32')
h5.create_dataset('x', data=params['x'])
h5.create_dataset('y', data=params['y'])
h5.close()

# def copy(fight_folders, output_train_fight_folder):
#     cmd_cp = "cp {} {}"
#     fight_subfolders = [dI for dI in glob.glob(
#     fight_folders) if os.path.isdir(os.path.join(fight_folders, dI))]

#     for idx, video_folder in enumerate(fight_subfolders):
#         print("{}/{}".format(idx, len(fight_subfolders)))
#         for img in glob.glob(video_folder+"/*png"):
#             prefix_video = os.path.basename(video_folder)
#             prefix_img = os.path.basename(img)
#             output_image = os.path.join(output_train_fight_folder, prefix_video+prefix_img)

#             subprocess.call(cmd_cp.format(img, output_image), shell=True)

# fight_folder = "/home/datasets/train/Fight/*"
# output_fight_folder = "/home/datasets/rwf-2000-frames/train/Fight/"

# copy(fight_folder, output_fight_folder)
