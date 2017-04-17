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

from itertools import izip
from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator
from common import *

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--input-pkl',
        type=str,
        help='pickle file with theta values')
    parser.add_argument('--motif-lens',
        type=str,
        help='comma-separated lengths of motifs (must all be odd)',
        default='3,5,7')

    args = parser.parse_args()

    return args

def main(args=sys.argv[1:]):

    args = parse_args()

    motif_len_vals = [int(m) for m in args.motif_lens.split(',')]
    for m in motif_len_vals:
        assert(m % 2 == 1)

    feat_generator = HierarchicalMotifFeatureGenerator(motif_lens=motif_len_vals)
    max_motif_len = max(motif_len_vals)
    full_motif_dict = feat_generator.feat_gens[-1].motif_dict

    # Load fitted theta file
    with open(args.input_pkl, "r") as f:
        theta = pickle.load(f)[0]

    # Combine the hierarchical thetas if that is the case
    full_theta = np.zeros((4**max_motif_len, theta.shape[1]))
    start_idx = 0
    for f in feat_generator.feat_gens[:len(motif_len_vals)]:
        motif_list = f.motif_list
        diff_len = max_motif_len - f.motif_len
        for m_idx, m in enumerate(motif_list):
            m_theta = theta[start_idx + m_idx,:]
            if diff_len == 0:
                full_m_idx = full_motif_dict[m]
                full_theta[full_m_idx,:] += m_theta
            else:
                flanks = itertools.product(["a", "c", "g", "t"], repeat=diff_len)
                for f in flanks:
                    full_m = "".join(f[:diff_len/2]) + m + "".join(f[diff_len/2:])
                    full_m_idx = full_motif_dict[full_m]
                    full_theta[full_m_idx,:] += m_theta
        start_idx += len(motif_list)

    theta_zipped = zip(full_theta.tolist(), feat_generator.feat_gens[-1].motif_list)
    if theta.shape[1] > 1:
        sort_lambda = lambda t: t[0][0]
    else:
        sort_lambda = lambda t: t[0][0]

    sorted_theta = sorted(theta_zipped, key=sort_lambda, reverse=True)

    known_hot_cold_regexs = compute_known_hot_and_cold(HOT_COLD_SPOT_REGS, max_motif_len)
    print "========== Top theta =========="
    for theta_tuple in sorted_theta[:10]:
        motif = theta_tuple[1]
        known_spot = print_known_cold_hot_spot(motif, known_hot_cold_regexs)
        print "%s %s\t: %s" % (motif, "(%s)" % known_spot if known_spot is not None else "\t", theta_tuple[0])

    print "========== Bottom theta =========="
    for theta_tuple in sorted_theta[-10:]:
        motif = theta_tuple[1]
        known_spot = print_known_cold_hot_spot(motif, known_hot_cold_regexs)
        print "%s %s\t: %s" % (motif, "(%s)" % known_spot if known_spot is not None else "\t", theta_tuple[0])

if __name__ == "__main__":
    main(sys.argv[1:])
