"""
Runs a bunch of parallel workers
Reads pickled files as input
Pickles results to an output file
"""
import sys
import argparse
import os
import pickle

import numpy as np

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--input-file',
        type=str,
        help='a pickle file',
        default='input.pkl')
    parser.add_argument('--output-file',
        type=str,
        help='a pickle file',
        default='output.pkl')

    parser.set_defaults()
    return parser.parse_args()

def main(args=sys.argv[1:]):
    args = parse_args()
    with open(args.input_file, "r") as input_file:
        workers = pickle.load(input_file)

    res = []
    for i, worker in enumerate(workers):
        res.append(worker.run())

    with open(args.output_file, "w") as output_file:
        pickle.dump(res, output_file)

if __name__ == "__main__":
    main(sys.argv[1:])
