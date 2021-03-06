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
from common import *

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--input-pkl',
        type=str,
        help='Pickle file with model parameters (theta values)',
        default='_output/context_model.pkl')
    parser.add_argument('--output-csv',
        type=str,
        help='place to output temporary CSV file',
        default='_output/out.csv')
    parser.add_argument('--output-pdf',
        type=str,
        help='PDF file to save output to',
        default='_output/out.pdf')
    parser.add_argument('--center-median',
        action='store_true',
        help="Should center theta parameters by median")
    parser.add_argument('--plot-separate',
        action='store_true',
        help="Plot hazard rates of different target nucleotides in separate PDFs")
    parser.add_argument('--no-conf-int',
        action='store_true',
        help="Remove confidence interval estimates")
    parser.add_argument('--center-nucs',
        type=str,
        default='A,T,G,C',
        help="Center nucleotides to plot")
    parser.add_argument('--y-lab',
        type=str,
        default='Aggregate Theta',
        help="y label of hedgehog plot")

    parser.set_defaults(no_conf_int=False)
    args = parser.parse_args()

    return args

def convert_to_csv(output_csv, theta_vals, theta_lower, theta_upper, full_feat_generator, per_target_model):
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
    agg_start_col = 1 if per_target_model else 0
    for col_idx in range(theta_vals.shape[1]):
        for motif, tval, tlow, tup in zip(padded_list, theta_vals[:, col_idx], theta_lower[:, col_idx], theta_upper[:, col_idx]):
            data.append([motif.upper(), col_idx+agg_start_col, tval, tlow, tup])

    with open(str(output_csv), 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

def plot_theta(output_csv, full_theta, theta_lower, theta_upper, output_pdf, per_target_model, full_feat_generator, max_motif_len, center_nucs, y_lab):
    if per_target_model:
        targets = 'A,C,G,T'
    else:
        targets = 'N'

    convert_to_csv(
        output_csv,
        full_theta,
        theta_lower,
        theta_upper,
        full_feat_generator,
        per_target_model,
    )

    # Call Rscript
    command = 'Rscript'
    script_file = 'R/create_bar_plot_from_file.R'

    cmd = [command, script_file, output_csv, str(max_motif_len), output_pdf, targets, center_nucs, y_lab]
    print "Calling:", " ".join(cmd)
    res = subprocess.call(cmd)

def main(args=sys.argv[1:]):
    args = parse_args()

    # Load fitted theta file
    with open(args.input_pkl, "r") as f:
        method_results = pickle.load(f)
        method_res = pick_best_model(method_results)
        per_target_model = method_res.refit_theta.shape[1] == NUM_NUCLEOTIDES + 1

    feat_generator = method_res.refit_feature_generator

    max_motif_len = feat_generator.motif_len
    full_feat_generator = HierarchicalMotifFeatureGenerator(
        motif_lens=[max_motif_len],
        left_motif_flank_len_list=feat_generator.left_motif_flank_len,
    )

    theta = method_res.refit_theta
    if args.center_median:
        theta -= np.median(theta)

    num_agg_cols = NUM_NUCLEOTIDES if per_target_model else 1
    agg_start_col = 1 if per_target_model else 0

    full_theta = np.zeros((full_feat_generator.feature_vec_len, num_agg_cols))
    theta_lower = np.zeros((full_feat_generator.feature_vec_len, num_agg_cols))
    theta_upper = np.zeros((full_feat_generator.feature_vec_len, num_agg_cols))
    for col_idx in range(num_agg_cols):
        full_theta[:,col_idx], theta_lower[:,col_idx], theta_upper[:,col_idx] = feat_generator.combine_thetas_and_get_conf_int(
            method_res.refit_theta,
            variance_est=method_res.variance_est,
            col_idx=col_idx + agg_start_col,
            add_targets=False,
        )

    agg_possible_motif_mask = full_feat_generator.get_possible_motifs_to_targets(full_theta.shape)
    full_theta[~agg_possible_motif_mask] = -np.inf
    theta_lower[~agg_possible_motif_mask] = -np.inf
    theta_upper[~agg_possible_motif_mask] = -np.inf

    if args.no_conf_int:
        theta_lower = full_theta
        theta_upper = full_theta

    plot_theta(args.output_csv, full_theta, theta_lower, theta_upper, args.output_pdf, per_target_model, full_feat_generator, max_motif_len, args.center_nucs, args.y_lab)

if __name__ == "__main__":
    main(sys.argv[1:])
