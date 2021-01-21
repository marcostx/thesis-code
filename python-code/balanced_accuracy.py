import numpy as np
import glob
import os 
from sklearn.metrics import balanced_accuracy_score

def get_violence_videos(file_inp, prediction_list):
    file_inp = open(file_inp)
    lines = file_inp.readlines()
    lines = [str.replace("\n","") for str in lines]
    video_names = [line_.split(" ")[0] for line_ in prediction_list]
    dict_labels = {}
    labels=[]
    preds = []
    # for line in lines:
    #     if line.split(" ")[2] in name_set:
    #         label = int(line.split(" ")[3])
    #         labels.append(label)
    for line in lines:
        dict_labels[line.split(" ")[2]] = int(line.split(" ")[3])

    for line in prediction_list:
        label = line.split(" ")[0]
        labels.append(dict_labels[label])
        if line.split(" ")[2] == 'f\n':
            preds.append(0)
        else:
            preds.append(1)

    return np.array(labels), np.array(preds)


def main():
    file_tagm = open("mediaeval_2015_test_predictions_raw_.txt", "r")
    labels_test = "mediaeval_stuffs/labels/violence_test.txt"

    lines = file_tagm.readlines()

    labels, preds = get_violence_videos(labels_test, lines)

    acc = balanced_accuracy_score(labels, preds)
    # print(preds[:1000])
    print("balanced accuracy : {}".format(acc))

    # pred_flow = np.zeros(4729)
    # y_flow = np.zeros(4729)
    # y_flow[:len(labels_test)] = 1

    # acc = balanced_accuracy_score(y_flow, pred_flow)
    # print("balanced accuracy : {}".format(acc))




if __name__ == '__main__':
    main()