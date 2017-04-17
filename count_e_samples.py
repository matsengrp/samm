"""
Given a pickled output file with theta values, convert to csv and plot bar charts
"""
import numpy as np
import subprocess
import sys
import argparse
import pickle
import csv
import itertools
from collections import Counter

from itertools import izip
from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--input-pkl',
        type=str,
        help='pickle file with e-samples')
    parser.add_argument('--motif-lens',
        type=str,
        help='comma-separated lengths of motifs (must all be odd)',
        default='3,5')

    args = parser.parse_args()

    return args

def main(args=sys.argv[1:]):
    args = parse_args()

    motif_len_vals = [int(m) for m in args.motif_lens.split(',')]
    for m in motif_len_vals:
        assert(m % 2 == 1)

    feat_generator = HierarchicalMotifFeatureGenerator(motif_lens=motif_len_vals)

    # Load fitted theta file
    with open(args.input_pkl, "r") as f:
        e_step_samples = pickle.load(f)

    # Count number of motifs just to track progress
    all_mutated_motifs = [[] for i in feat_generator.motif_lens]
    for order_sample in e_step_samples:
        mutated_motifs = feat_generator.count_mutated_motifs(order_sample)
        for i, m_list in enumerate(mutated_motifs):
            all_mutated_motifs[i] += m_list

    for m_list in all_mutated_motifs:
        c = Counter(m_list)
        print c.most_common(10)

if __name__ == "__main__":
    main(sys.argv[1:])
