# License: MIT. See LICENSE.txt for the full license.
# DISTRIBUTION STATEMENT A. Approved for public release: distribution unlimited.

# © 2017 Massachusetts Institute of Technology.
#
# MIT Proprietary, Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or
# 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are
# defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than
# as specifically authorized by the U.S. Government may violate any copyrights that exist in this
# work.

import h5py
import argparse
import numpy as np
import os
from os.path import join
from sklearn.model_selection import StratifiedKFold

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

def h5_to_npy(inp, output_file, target, dtype):
    """ Save array stored in hdf5 dataset to a .npy file: {dest_dir}/{dest_name}.npy
     Parameters
        ----------
        inp : str
            Path to the directory where .h5 file is located.
        output : str
            Path to the directory in which the .npy file will be saved.
        dtype : numpy.dtype
            Data type of the array being saved.
        Returns
        -------
        None
    """
    print(inp)
    h5_file = h5py.File(inp, 'r')
    print('Saving {} as npy ...'.format(inp))
    np.save(output_file, np.asarray(h5_file[target]))
    print('Saved !!')

def holdout_dataset():
    inp = "/home/src/mediaeval-train-efficientnet-flow.h5"
    out_x = "/home/src/train-flow-media_x.npy"
    out_y = "/home/src/train-flow-media_y.npy"

    h5_to_npy(inp, out_x, 'x', np.float32)
    h5_to_npy(inp, out_y, 'y', np.float32)

    inp = "/home/src/mediaeval-test-efficientnet-flow.h5"
    out_x = "/home/src/val-flow-media_x.npy"
    out_y = "/home/src/val-flow-media_y.npy"

    h5_to_npy(inp, out_x, 'x', np.float32)
    h5_to_npy(inp, out_y, 'y', np.float32)

    print("finished")

def cv_dataset():
    inp = "/home/src/hockey-efficientnet.h5"
     # whole dataset
    train_h5 = h5py.File(inp, 'r')
    # #
    X, y = train_h5['x'], train_h5['y']

    # get splits
    skf = StratifiedKFold(n_splits=5)
    index = 0

    for train_index, test_index in skf.split(X, y):
        trainx, testx = X[train_index], X[test_index]
        trainy, testy = y[train_index], y[test_index]

        out_x = "/home/src/{}/train-hockey_x.npy".format(index)
        out_y = "/home/src/{}/train-hockey_y.npy".format(index)

        np.save(out_x, np.asarray(trainx))
        np.save(out_y, np.asarray(trainy))

        inp = "/home/src/rwf-flow-val-finetuned.h5"
        out_x = "/home/src/{}/test-hockey_x.npy".format(index)
        out_y = "/home/src/{}/test-hockey_y.npy".format(index)

        np.save(out_x, np.asarray(testx))
        np.save(out_y, np.asarray(testy))
        index+=1

    print("finished")


if __name__ == '__main__':
    holdout_dataset()
