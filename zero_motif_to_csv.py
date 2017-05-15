"""
Given a pickled output file with theta values, convert to a zero motif csv file
"""
import os
import numpy as np
import sys
import argparse
import pickle
import csv

from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator
from common import ZERO_THRES

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--input-pkl',
        type=str,
        help='pickle file with theta values')
    parser.add_argument('--output-csv',
        type=str,
        help='where to put csv output file - will be in the same folder',
        default='zero_motifs.csv')
    parser.add_argument('--motif-lens',
        type=str,
        help='comma-separated lengths of motifs (must all be odd)',
        default='3,5,7')
    parser.add_argument('--positions-mutating',
        type=str,
        help='comma-separated mut pos',
        default=None)

    args = parser.parse_args()
    args.output_path = os.path.dirname(os.path.realpath(args.input_pkl))
    args.output_csv = os.path.join(args.output_path, args.output_csv)
    print "Output csv file", args.output_csv
    return args

def main(args=sys.argv[1:]):
    args = parse_args()

    args.motif_len_vals = [int(m) for m in args.motif_lens.split(',')]
    for m in args.motif_len_vals:
        assert(m % 2 == 1)

    if args.positions_mutating is not None:
        args.positions_mutating = [[int(m) for m in positions.split(',')] for positions in args.positions_mutating.split(':')]
        for motif_len, positions in zip(args.motif_lens, args.positions_mutating):
            for m in positions:
                assert(m in range(motif_len))

    feat_generator = HierarchicalMotifFeatureGenerator(
            motif_lens=args.motif_len_vals,
            left_motif_flank_len_list=args.positions_mutating,
        )
    args.max_motif_len = max(args.motif_len_vals)

    # Load fitted theta file
    with open(args.input_pkl, "r") as f:
        theta = pickle.load(f)[0]

    # Construct the lines in the csv file
    csv_lines = []
    for i in range(theta.shape[0]):
        motif = feat_generator.motif_list[i]
        csv_line = [motif, feat_generator.left_motif_flank_len]
        for j in range(theta.shape[1]):
            theta_val = theta[i,j]
            if theta_val == -np.inf:
                # If motif is impossible, set 0 for the zero mask
                csv_line.append(0)
            elif np.abs(theta_val) < ZERO_THRES:
                # If motif is zero, set 1 for the zero mask
                csv_line.append(1)
            else:
                # If motif is not zero, set 0 for the zero mask
                csv_line.append(0)
        csv_lines.append(csv_line)

    # Write csv lines to file now
    with open(args.output_csv, "w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(csv_lines)

if __name__ == "__main__":
    main(sys.argv[1:])
