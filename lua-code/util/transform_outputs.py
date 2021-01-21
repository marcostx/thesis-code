import argparse
import sys

import numpy as np


def translate(value, OldMin, OldMax, NewMin, NewMax):
	OldRange = (OldMax - OldMin)
	NewRange = (NewMax - NewMin)
	NewValue = (((value - OldMin) * NewRange) / OldRange) + NewMin
	return NewValue

def parse_info(parser):
    parser.add_argument("-i", dest='input_file', default='mediaeval_test_predictions_inceptv3.txt')
    parser.add_argument("-o", dest='output_file', default='mediaeval_test_predictions_correct_incept.txt')
    args, _ = parser.parse_known_args(argv)

    return args

if __name__ == "__main__":
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    args = parse_info(parser)
    output_data=[]

    file = open(args.input_file)
    outfile = open(args.output_file, "w")

    lines = file.readlines()

    lines = [string.replace("\n","") for string in lines]

    values =[float(line.split(" ")[1]) for line in lines]
    for idx, line in enumerate(lines):
    	line_transformed = line.split(" ")[0]+" "+str(translate(values[idx], min(values), 0.0, 0.0, 1.0 ))+" "+line.split(" ")[2]+"\n"
    	outfile.write(line_transformed)
    outfile.close()
