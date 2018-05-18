"""
Given a pickled output file with theta values, plot bar charts
"""
import numpy as np
import subprocess
import sys
import argparse
import pickle
import csv

from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator
from compare_simulated_shazam_vs_samm import ShazamModel
from common import *

from plot_samm import plot_theta

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--mut',
        type=str,
        default=None,
        help='mutabilities csv')
    parser.add_argument('--sub',
        type=str,
        default=None,
        help='substitution csv. only include this file for per-target')
    parser.add_argument('--output-csv',
        type=str,
        help='place to output temporary CSV file',
        default='_output/out.csv')
    parser.add_argument('--output-pdf',
        type=str,
        help='PDF file to save output to',
        default='_output/out.pdf')

    args = parser.parse_args()

    return args

def convert_to_csv(output_csv, theta_vals, theta_lower, theta_upper, full_feat_generator):
    """
    Take pickle file and convert to csv for use in R
    """
    padded_list = list(full_feat_generator.motif_list)

    if full_feat_generator.num_feat_gens > 1:
        # pad offset motifs with Ns
        # if center base is mutating this will just yield usual motif list
        for idx, (motif, mut_pos) in enumerate(zip(padded_list, full_feat_generator.mutating_pos_list)):
            left_pad = (full_feat_generator.max_left_motif_flank_len - mut_pos) * 'n'
            right_pad = (mut_pos - full_feat_generator.max_right_motif_flank_len + full_feat_generator.motif_len - 1) * 'n'
            padded_list[idx] = left_pad + motif + right_pad

    header = ['motif', 'target', 'theta', 'theta_lower', 'theta_upper']
    data = []
    for col_idx in range(theta_vals.shape[1]):
        for motif, tval, tlow, tup in zip(padded_list, theta_vals[:, col_idx], theta_lower[:, col_idx], theta_upper[:, col_idx]):
            data.append([motif.upper(), col_idx, tval, tlow, tup])

    with open(str(output_csv), 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

def main(args=sys.argv[1:]):
    args = parse_args()

    MOTIF_LEN = 5

    full_feat_generator = HierarchicalMotifFeatureGenerator(
        motif_lens=[MOTIF_LEN],
        left_motif_flank_len_list=[[MOTIF_LEN/2]],
    )

    # Load fitted theta file
    shazam_model = ShazamModel(MOTIF_LEN, args.mut, args.sub)

    full_theta = shazam_model.agg_refit_theta
    # center median
    theta_med = np.median(full_theta[~np.isinf(full_theta)])
    full_theta -= theta_med
    theta_lower = full_theta
    theta_upper = full_theta

    if full_theta.shape[1] > 1:
        plot_theta(args.output_csv, full_theta, theta_lower, theta_upper, args.output_pdf, 'A,C,G,T', full_feat_generator, MOTIF_LEN)
    else:
        plot_theta(args.output_csv, full_theta, theta_lower, theta_upper, args.output_pdf, 'N', full_feat_generator, MOTIF_LEN)

if __name__ == "__main__":
    main(sys.argv[1:])
