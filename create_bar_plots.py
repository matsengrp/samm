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

def convert_to_csv(target, theta_vals, motif_list, theta_lower, theta_upper, motif_lens, mutating_pos_list, feat_generator, num_mut_pos):
    """
    Take pickle file and convert to csv for use in R
    """
    padded_list = list(motif_list* num_mut_pos)

    if num_mut_pos > 1:
        # pad offset motifs with Ns
        for idx, (motif, mut_pos) in enumerate(zip(padded_list, mutating_pos_list)):
            left_pad = (feat_generator.max_left_motif_flank_len - mut_pos) * 'n'
            right_pad = (mut_pos - feat_generator.max_right_motif_flank_len + feat_generator.motif_len - 1) * 'n'
            padded_list[idx] = left_pad + motif + right_pad

    header = ['motif', 'target', 'theta', 'theta_lower', 'theta_upper']
    data = []
    for col_idx in range(theta_vals.shape[1]):
        for motif, tval, tlow, tup in zip(padded_list, theta_vals[:, col_idx], theta_lower[:, col_idx], theta_upper[:, col_idx]):
            data.append([motif.upper(), col_idx, tval, tlow, tup])

    with open(str(target), 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

def get_theta_conf_int(args, feat_generator, full_feat_generator, theta, covariance_est, col_idx, num_mut_pos):
    full_theta_size = (4**args.max_motif_len) * (num_mut_pos)
    full_theta = np.zeros(full_theta_size)
    theta_lower = np.zeros(full_theta_size)
    theta_upper = np.zeros(full_theta_size)

    if len(args.motif_len_vals) > 1 and num_mut_pos == 1:
        # Combine the hierarchical thetas if that is the case
        theta_index_matches = {i:[] for i in range(full_theta_size)}

        start_idx = 0
        for feat_gen in feat_generator.feat_gens[:len(args.motif_len_vals)]:
            motif_list = feat_gen.motif_list
            for m_idx, m in enumerate(motif_list):

                raw_theta_idx = start_idx + m_idx
                m_theta = theta[raw_theta_idx, 0]
                if col_idx != 0:
                    m_theta += theta[raw_theta_idx, col_idx]

                if feat_gen.hier_offset == 0:
                    full_m_idx = full_feat_generator.motif_dict[m]
                    full_theta[full_m_idx] += m_theta

                    theta_index_matches[full_m_idx].append(raw_theta_idx)
                    if col_idx != 0:
                        theta_index_matches[full_m_idx].append(raw_theta_idx + col_idx * theta.shape[0])
                else:
                    flanks = itertools.product(["a", "c", "g", "t"], repeat=2*feat_gen.hier_offset)
                    for f in flanks:
                        # assume for now hierarchical will just have center mutating
                        full_m = "".join(f[:feat_gen.hier_offset]) + m + "".join(f[feat_gen.hier_offset:])
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

    elif num_mut_pos > 1 and len(args.motif_len_vals) == 1:
        # Combine the hierarchical thetas if that is the case
        theta_index_matches = {i:[] for i in range(full_theta_size)}

        start_idx = 0
        for feat_gen in feat_generator.feat_gens[:num_mut_pos]:
            motif_list = feat_gen.motif_list
            for m_idx, m in enumerate(motif_list):

                raw_theta_idx = start_idx + m_idx
                m_theta = theta[raw_theta_idx, 0]
                if col_idx != 0:
                    if np.isfinite(theta[raw_theta_idx, col_idx]):
                        m_theta += theta[raw_theta_idx, col_idx]
                    else:
                        m_theta = 0

                full_m_idx = start_idx + full_feat_generator.motif_dict[m]
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

def plot_theta(args, full_theta, motif_list, theta_lower, theta_upper, output_pdf, mutating_pos_list, targets, feat_generator, num_mut_pos):
    convert_to_csv(
        args.output_csv,
        full_theta,
        motif_list,
        theta_lower,
        theta_upper,
        [args.max_motif_len],
        mutating_pos_list,
        feat_generator,
        num_mut_pos
    )

    # Call Rscript
    command = 'Rscript'
    script_file = 'R/create_bar_plot_from_file.R'

    cmd = [command, script_file, args.output_csv, str(args.max_motif_len), output_pdf, targets]
    print "Calling:", " ".join(cmd)
    res = subprocess.call(cmd)

def main(args=sys.argv[1:]):

    args = parse_args()

    args.motif_len_vals = [int(m) for m in args.motif_lens.split(',')]
    for m in args.motif_len_vals:
        assert(m % 2 == 1)
        
    if args.positions_mutating is None:
        # default to central base mutating
        args.max_left_flank = None
        args.max_right_flank = None
        args.positions_mutating = [[m/2] for m in args.motif_len_vals]
    else:
        args.positions_mutating = [[int(m) for m in positions.split(',')] for positions in args.positions_mutating.split(':')]
        for motif_len, positions in zip(args.motif_len_vals, args.positions_mutating):
            for m in positions:
                assert(m in range(motif_len))

    args.max_motif_len = max(args.motif_len_vals)

    motifs_to_remove, target_pairs_to_remove = read_zero_motif_csv(args.zero_motifs, args.per_target_model)
    feat_generator = HierarchicalMotifFeatureGenerator(
        motif_lens=args.motif_len_vals,
        motifs_to_remove=motifs_to_remove,
        left_motif_flank_len_list=args.positions_mutating,
    )
    mutating_pos_list = feat_generator.mutating_pos_list
    # number of offsets
    num_mut_pos = feat_generator.num_feat_gens / len(args.motif_len_vals)

    full_feat_generator = HierarchicalMotifFeatureGenerator(
        motif_lens=[args.max_motif_len],
    )
    motif_list = full_feat_generator.motif_list

    # Load fitted theta file
    with open(args.input_pkl, "r") as f:
        pickled_tup = pickle.load(f)
        theta = pickled_tup[0]
        if args.center_median:
            theta -= np.median(theta)

        if args.no_conf_int or pickled_tup[2] is None:
            covariance_est = np.zeros((theta.size, theta.size))
        else:
            covariance_est = pickled_tup[2]
        assert(theta.shape[0] == feat_generator.feature_vec_len)

    # with open("_output/fisher_info_obs.pkl", "r") as f:
    #     fisher_info1 = pickle.load(f)
    #
    #     possible_theta_mask = get_possible_motifs_to_targets(feat_generator.motif_list, theta.shape, feat_generator.mutating_pos_list)
    #     zero_theta_mask = get_zero_theta_mask(target_pairs_to_remove, feat_generator, theta.shape)
    #     theta_mask = possible_theta_mask & ~zero_theta_mask
    #     theta_mask_flat = theta_mask.reshape((theta_mask.size,), order="F")
    #
    #     sample_obs_information = (fisher_info1[theta_mask_flat,:])[:,theta_mask_flat]
    #     covariance_est = np.linalg.inv(sample_obs_information)

    if len(args.motif_len_vals) > 1:
        # hierarchical model combines values
        full_theta = np.zeros((len(motif_list), theta.shape[1]))
        theta_lower = np.zeros((len(motif_list), theta.shape[1]))
        theta_upper = np.zeros((len(motif_list), theta.shape[1]))
    else:
        # otherwise just use shape of theta
        full_theta = np.zeros(theta.shape)
        theta_lower = np.zeros(theta.shape)
        theta_upper = np.zeros(theta.shape)

    for col_idx in range(theta.shape[1]):
        full_theta[:,col_idx], theta_lower[:,col_idx], theta_upper[:,col_idx] = \
                get_theta_conf_int(args, feat_generator, full_feat_generator, theta, covariance_est, col_idx, num_mut_pos)

    if args.per_target_model:
        if args.plot_separate:
            for col_idx, target in enumerate(['N', 'A', 'C', 'G', 'T']):
                output_pdf = args.output_pdf.replace(".pdf", "_col%d.pdf" % col_idx)
                plot_theta(args, full_theta, motif_list, theta_lower, theta_upper, output_pdf, mutating_pos_list, target, feat_generator, num_mut_pos)
        else:
            plot_theta(args, full_theta, motif_list, theta_lower, theta_upper, args.output_pdf, mutating_pos_list, 'N,A,C,G,T', feat_generator, num_mut_pos)
    else:
        plot_theta(args, full_theta, motif_list, theta_lower, theta_upper, args.output_pdf, mutating_pos_list, 'N', feat_generator, num_mut_pos)

if __name__ == "__main__":
    main(sys.argv[1:])
