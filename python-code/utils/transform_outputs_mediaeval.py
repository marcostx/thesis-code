import argparse
import sys

import numpy as np


def translate(value, OldMin, OldMax, NewMin, NewMax):
    OldRange = (OldMax - OldMin)
    NewRange = (NewMax - NewMin)
    NewValue = (((value - OldMin) * NewRange) / OldRange) + NewMin
    return NewValue

def log_to_normal(value):
    return 2**value
    

def parse_info(parser):
    parser.add_argument("-i", "--input_file", dest='input_file',
                        default='/home/marcostx/master-degree/temporal-attention-violence-detection/mediaeval_test_predictions_flow.txt')
    parser.add_argument("-o","--output_file", dest='output_file',
                        default='/home/marcostx/master-degree/temporal-attention-violence-detection/mediaeval_test_predictions_flow_.txt')
    args, _ = parser.parse_known_args(argv)

    return args


if __name__ == "__main__":
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    args = parse_info(parser)
    output_data = []

    file = open(args.input_file)
    outfile = open(args.output_file, "w")

    lines = file.readlines()

    lines = [str_.replace("\n", "") for str_ in lines]

    values = [float(line.split(" ")[1]) for line in lines]
    for idx, line in enumerate(lines):
        # line_transformed = line.split(" ")[
        #     0]+" "+str(translate(values[idx], min(values), 0.0, 0.0, 1.0))+" "+line.split(" ")[2]+"\n"
        line_transformed = line.split(" ")[0]+" "+str(log_to_normal(values[idx]))+" "+line.split(" ")[2]+"\n"
        outfile.write(line_transformed)
    outfile.close()
