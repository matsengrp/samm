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

from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator
from read_data import read_zero_motif_csv
from common import *

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--input-pkl',
        type=str,
        help='pickle file with theta values')
    parser.add_argument('--output-csv',
        type=str,
        help='where to put csv output file',
        default='_output/out.csv')
    parser.add_argument('--zero-motifs',
        type=str,
        help='csv file indicating which motifs are zero',
        default='')
    parser.add_argument('--motif-lens',
        type=str,
        help='comma-separated lengths of motifs (must all be odd)',
        default='3,5,7')
    parser.add_argument('--positions-mutating',
        type=str,
        help="""
        length of left motif flank determining which position is mutating; comma-separated within
        a motif length, colon-separated between, e.g., --motif-lens 3,5 --left-flank-lens 0,1:0,1,2 will
        be a 3mer with first and second mutating position and 5mer with first, second and third
        """,
        default=None)
    parser.add_argument('--output-pdf',
        type=str,
        help='pdf file to save output to',
        default='_output/out.pdf')
    parser.add_argument('--per-target-model',
        action='store_true')
    parser.add_argument('--center-median',
        action='store_true')
    parser.add_argument('--no-conf-int',
        action='store_true')
    parser.add_argument('--plot-separate',
        action='store_true')

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

def plot_theta(output_csv, full_theta, theta_lower, theta_upper, output_pdf, targets, full_feat_generator, max_motif_len):
    convert_to_csv(
        output_csv,
        full_theta,
        theta_lower,
        theta_upper,
        full_feat_generator,
    )

    # Call Rscript
    command = 'Rscript'
    script_file = 'R/create_bar_plot_from_file.R'

    cmd = [command, script_file, output_csv, str(max_motif_len), output_pdf, targets]
    print "Calling:", " ".join(cmd)
    res = subprocess.call(cmd)

def main(args=sys.argv[1:]):

    args = parse_args()

    args.motif_len_vals = [int(m) for m in args.motif_lens.split(',')]

    args.max_motif_len = max(args.motif_len_vals)

    args.positions_mutating, args.max_mut_pos = process_mutating_positions(args.motif_len_vals, args.positions_mutating)

    full_feat_generator = HierarchicalMotifFeatureGenerator(
        motif_lens=[args.max_motif_len],
        left_motif_flank_len_list=args.max_mut_pos,
    )

    # Load fitted theta file
    with open(args.input_pkl, "r") as f:
        method_results = pickle.load(f)
        method_res = pick_best_model(method_results)

    theta = method_res.refit_theta
    if args.center_median:
        theta -= np.median(theta)

    covariance_est = method_res.variance_est

    feat_generator = HierarchicalMotifFeatureGenerator(
        motif_lens=args.motif_len_vals,
        feats_to_remove=method_res.model_masks.feats_to_remove,
        left_motif_flank_len_list=args.positions_mutating,
    )
    mutating_pos_list = feat_generator.mutating_pos_list

    full_theta = np.zeros((full_feat_generator.feature_vec_len, theta.shape[1]))
    theta_lower = np.zeros((full_feat_generator.feature_vec_len, theta.shape[1]))
    theta_upper = np.zeros((full_feat_generator.feature_vec_len, theta.shape[1]))
    for col_idx in range(theta.shape[1]):
        full_theta[:,col_idx], theta_lower[:,col_idx], theta_upper[:,col_idx] = \
                combine_thetas_and_get_conf_int(feat_generator, full_feat_generator, method_res, col_idx)

    if args.per_target_model:
        if args.plot_separate:
            for col_idx, target in enumerate(['N', 'A', 'C', 'G', 'T']):
                output_pdf = args.output_pdf.replace(".pdf", "_col%d.pdf" % col_idx)
                plot_theta(args.output_csv, full_theta, theta_lower, theta_upper, output_pdf, target, full_feat_generator, args.max_motif_len)
        else:
            plot_theta(args.output_csv, full_theta, theta_lower, theta_upper, args.output_pdf, 'N,A,C,G,T', full_feat_generator, args.max_motif_len)
    else:
        plot_theta(args.output_csv, full_theta, theta_lower, theta_upper, args.output_pdf, 'N', full_feat_generator, args.max_motif_len)

if __name__ == "__main__":
    main(sys.argv[1:])
