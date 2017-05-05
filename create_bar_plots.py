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
        help='where to put csv output file',
        default='_output/zero_motifs.csv')
    parser.add_argument('--motif-lens',
        type=str,
        help='comma-separated lengths of motifs (must all be odd)',
        default='3,5,7')
    parser.add_argument('--output-pdf',
        type=str,
        help='pdf file to save output to',
        default='_output/out.pdf')
    parser.add_argument('--per-target-model',
        action='store_true')


    args = parser.parse_args()

    return args

def convert_to_csv(target, theta_vals, theta_lower, theta_upper, motif_lens):
    """
    Take pickle file and convert to csv for use in R
    """
    feat_generator = HierarchicalMotifFeatureGenerator(motif_lens=motif_lens)
    motif_list = feat_generator.motif_list

    with open(str(target), 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(izip(
            [motif.upper() for motif in motif_list],
            theta_vals.ravel(),
            theta_lower.ravel(),
            theta_upper.ravel(),
        ))

def get_theta_conf_int(args, feat_generator, full_feat_generator, theta, covariance_est, col_idx):
    full_theta_size = 4**args.max_motif_len
    full_theta = np.zeros(full_theta_size)
    theta_lower = np.zeros(full_theta_size)
    theta_upper = np.zeros(full_theta_size)

    if len(args.motif_len_vals) > 1:
        # Combine the hierarchical thetas if that is the case
        theta_index_matches = {i:[] for i in range(full_theta_size)}

        start_idx = 0
        for f in feat_generator.feat_gens[:len(args.motif_len_vals)]:
            motif_list = f.motif_list
            diff_len = args.max_motif_len - f.motif_len
            for m_idx, m in enumerate(motif_list):
                raw_theta_idx = start_idx + m_idx
                m_theta = theta[raw_theta_idx, 0]
                if col_idx != 0:
                    m_theta += theta[raw_theta_idx, col_idx]

                if diff_len == 0:
                    full_m_idx = full_feat_generator.motif_dict[m]
                    full_theta[full_m_idx] += m_theta

                    theta_index_matches[full_m_idx].append(raw_theta_idx)
                    if col_idx != 0:
                        theta_index_matches[full_m_idx].append(raw_theta_idx + col_idx * theta.shape[0])
                else:
                    flanks = itertools.product(["a", "c", "g", "t"], repeat=diff_len)
                    for f in flanks:
                        full_m = "".join(f[:diff_len/2]) + m + "".join(f[diff_len/2:])
                        full_m_idx = full_feat_generator.motif_dict[full_m]
                        full_theta[full_m_idx] += m_theta

                        theta_index_matches[full_m_idx].append(raw_theta_idx)
                        if col_idx != 0:
                            theta_index_matches[full_m_idx].append(raw_theta_idx + col_idx * theta.shape[0])

            start_idx += len(motif_list)

        for full_theta_idx, matches in theta_index_matches.iteritems():
            var_est = 0
            for i in matches:
                for j in matches:
                    var_est += covariance_est[i,j]

            standard_err_est = np.sqrt(var_est)
            theta_lower[full_theta_idx] = full_theta[full_theta_idx] - ZSCORE_95 * standard_err_est
            theta_upper[full_theta_idx] = full_theta[full_theta_idx] + ZSCORE_95 * standard_err_est
    else:
        for i, m in enumerate(full_feat_generator.motif_list):
            if m in feat_generator.motif_dict:
                theta_idx = feat_generator.motif_dict[m]
                full_theta[i] = theta[theta_idx]
                standard_err_est = np.sqrt(covariance_est[theta_idx, theta_idx])
                theta_lower[i] = theta[theta_idx] - ZSCORE_95 * standard_err_est
                theta_upper[i] = theta[theta_idx] + ZSCORE_95 * standard_err_est
    return full_theta, theta_lower, theta_upper

def plot_theta(args, full_theta, theta_lower, theta_upper, output_pdf):
    convert_to_csv(args.output_csv, full_theta, theta_lower, theta_upper, [args.max_motif_len])

    # Call Rscript
    command = 'Rscript'
    script_file = 'R/create_bar_plot_from_file.R'

    cmd = [command, script_file, args.output_csv, str(args.max_motif_len), output_pdf]
    print "Calling:", " ".join(cmd)
    res = subprocess.call(cmd)

def main(args=sys.argv[1:]):

    args = parse_args()

    args.motif_len_vals = [int(m) for m in args.motif_lens.split(',')]
    for m in args.motif_len_vals:
        assert(m % 2 == 1)
    args.max_motif_len = max(args.motif_len_vals)

    motifs_to_remove, target_pairs_to_remove = read_zero_motif_csv(args.zero_motifs, args.per_target_model)
    feat_generator = HierarchicalMotifFeatureGenerator(
        motif_lens=args.motif_len_vals,
        motifs_to_remove=motifs_to_remove,
    )
    full_feat_generator = HierarchicalMotifFeatureGenerator(
        motif_lens=[args.max_motif_len],
    )

    # Load fitted theta file
    with open(args.input_pkl, "r") as f:
        theta = pickle.load(f)[0]
        covariance_est = pickle.load(f)[2]
        assert(theta.shape[0] == feat_generator.feature_vec_len)

    # with open("_output/fisher_info_obs.pkl", "r") as f:
    #     fisher_info1 = pickle.load(f)
    #
    #     possible_theta_mask = get_possible_motifs_to_targets(feat_generator.motif_list, theta.shape)
    #     zero_theta_mask = get_zero_theta_mask(target_pairs_to_remove, feat_generator, theta.shape)
    #     theta_mask = possible_theta_mask & ~zero_theta_mask
    #     theta_mask_flat = theta_mask.reshape((theta_mask.size,), order="F")
    #
    #     sample_obs_information = (fisher_info1[theta_mask_flat,:])[:,theta_mask_flat]
    #     covariance_est = np.linalg.inv(sample_obs_information)

    for col_idx in range(theta.shape[1]):
        output_pdf = args.output_pdf.replace(".pdf", "_col%d.pdf" % col_idx)
        full_theta, theta_lower, theta_upper = get_theta_conf_int(args, feat_generator, full_feat_generator, theta, covariance_est, col_idx)
        plot_theta(args, full_theta, theta_lower, theta_upper, output_pdf)

if __name__ == "__main__":
    main(sys.argv[1:])
