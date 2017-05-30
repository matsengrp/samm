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
    parser.add_argument('--most-nonzero',
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

def combine_thetas_and_get_conf_int(feat_generator, full_feat_generator, theta_flat, covariance_est, col_idx, z_stat, feat_tuple):
    """
    Combine hierarchical and offset theta values
    """
    full_theta_size = full_feat_generator.feature_vec_len
    full_theta = np.zeros(full_theta_size)
    theta_lower = np.zeros(full_theta_size)
    theta_upper = np.zeros(full_theta_size)

    theta_index_matches = {i:[] for i in range(full_theta_size)}

    for i, feat_gen in enumerate(feat_generator.feat_gens):
        for m_idx, m in enumerate(feat_gen.motif_list):
            if (m, feat_gen.left_motif_flank_len, 0) in feat_tuple:
                mut_theta_idx = feat_tuple.index((m, feat_gen.left_motif_flank_len, 0))
                m_theta = theta_flat[mut_theta_idx]
            else:
                mut_theta_idx = None
                m_theta = 0

            if col_idx != 0 and (m, feat_gen.left_motif_flank_len, col_idx) in feat_tuple:
                raw_theta_idx = feat_tuple.index((m, feat_gen.left_motif_flank_len, col_idx))
                m_theta += theta_flat[raw_theta_idx]
            else:
                if mut_theta_idx is None or NUCLEOTIDE_DICT[m[feat_gen.left_motif_flank_len]] == col_idx - 1:
                    continue
                else:
                    raw_theta_idx = None

            if feat_gen.motif_len == full_feat_generator.motif_len:
                # Already at maximum motif length, so nothing to combine
                full_m_idx = full_feat_generator.motif_dict[m][feat_gen.left_motif_flank_len]
                full_theta[full_m_idx] += m_theta

                if mut_theta_idx is not None:
                    theta_index_matches[full_m_idx].append(mut_theta_idx)
                if raw_theta_idx is not None:
                    theta_index_matches[full_m_idx].append(raw_theta_idx)
            else:
                # Combine hierarchical feat_gens for given left_motif_len
                for full_feat_gen in full_feat_generator.feat_gens:
                    flanks = itertools.product(["a", "c", "g", "t"], repeat=full_feat_gen.motif_len - feat_gen.motif_len)
                    for f in flanks:
                        full_m = "".join(f[:feat_gen.hier_offset]) + m + "".join(f[feat_gen.hier_offset:])
                        full_m_idx = full_feat_generator.motif_dict[full_m][full_feat_gen.left_motif_flank_len]
                        full_theta[full_m_idx] += m_theta

                        theta_index_matches[full_m_idx].append(raw_theta_idx)
                        if col_idx != 0:
                            theta_index_matches[full_m_idx].append(raw_theta_idx + col_idx * theta.shape[0])

    for full_theta_idx, matches in theta_index_matches.iteritems():
        var_est = 0
        for i, j in zip(matches, matches):
            var_est += covariance_est[i,j]

        standard_err_est = np.sqrt(var_est)
        theta_lower[full_theta_idx] = full_theta[full_theta_idx] - z_stat * standard_err_est
        theta_upper[full_theta_idx] = full_theta[full_theta_idx] + z_stat * standard_err_est

    return full_theta, theta_lower, theta_upper

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
    for m in args.motif_len_vals:
        assert(m % 2 == 1)
        
    args.max_motif_len = max(args.motif_len_vals)

    if args.positions_mutating is None:
        # default to central base mutating
        args.positions_mutating = [[m/2] for m in args.motif_len_vals]
        args.max_mut_pos = [[args.max_motif_len/2]]
    else:
        args.positions_mutating = [[int(m) for m in positions.split(',')] for positions in args.positions_mutating.split(':')]
        for motif_len, positions in zip(args.motif_len_vals, args.positions_mutating):
            for m in positions:
                assert(m in range(motif_len))
        args.max_mut_pos = [mut_pos for mut_pos, motif_len in zip(args.positions_mutating, args.motif_len_vals) if motif_len == args.max_motif_len]

    # Load fitted theta file
    with open(args.input_pkl, "r") as f:
        pickled_tup = pickle.load(f)

    # pick "best" model
    if args.most_nonzero:
        pickled_tup = [pickled_elt for pickled_elt in pickled_tup if pickled_elt.has_refit_data and pickled_elt.variance_est is not None]
        nonzero_cross_list = [(idx, m.num_not_crossing_zero) for idx, m in enumerate(pickled_tup) \
                if np.abs(m.variance_est).max() < 10.]
        if nonzero_cross_list:
            best_model_tup = max(nonzero_cross_list, key=lambda item: item[1])
            best_model_idx = nonzero_cross_list[nonzero_cross_list.index(best_model_tup)][0]
            best_model = pickled_tup[best_model_idx]
        else:
            raise ValueError('Error: no model has refit data')
    else:
        ll_list = [(idx, m.log_lik_ratio) for idx, m in enumerate(pickled_tup) if m.has_refit_data and m.log_lik_ratio is not None]
        if ll_list:
            best_model_tup = min(ll_list, key=lambda item: item[1])
            best_model_idx = ll_list[ll_list.index(best_model_tup)][0]
            best_model = pickled_tup[best_model_idx]
        else:
            raise ValueError('Error: no model has refit data')

    theta = best_model.refit_theta
    theta_mask = best_model.refit_possible_theta_mask & ~best_model.model_masks.zero_theta_mask_refit
    theta_mask_flat = theta_mask.reshape((theta_mask.size,), order="C")
    theta_flat = theta.reshape((theta.size,), order="C")
    theta_flat = theta_flat[theta_mask_flat]

    if args.center_median:
        theta -= np.median(theta)

    if args.no_conf_int or best_model.variance_est is None:
        covariance_est = np.zeros((theta_flat.size, theta_flat.size))
    else:
        covariance_est = best_model.variance_est

    full_feat_generator = HierarchicalMotifFeatureGenerator(
        motif_lens=[args.max_motif_len],
        left_motif_flank_len_list=args.max_mut_pos,
    )

    feat_generator = HierarchicalMotifFeatureGenerator(
        motif_lens=args.motif_len_vals,
        feats_to_remove=best_model.model_masks.feats_to_remove,
        left_motif_flank_len_list=args.positions_mutating,
    )

    full_theta = np.zeros((full_feat_generator.feature_vec_len, theta.shape[1]))
    theta_lower = np.zeros((full_feat_generator.feature_vec_len, theta.shape[1]))
    theta_upper = np.zeros((full_feat_generator.feature_vec_len, theta.shape[1]))

    theta_mask = best_model.refit_possible_theta_mask & ~best_model.model_masks.zero_theta_mask_refit
    feat_tups = []
    for i in range(best_model.model_masks.zero_theta_mask_refit.shape[0]):
        motif = feat_generator.motif_list[i]
        mut_pos = feat_generator.mutating_pos_list[i]
        for j in range(best_model.model_masks.zero_theta_mask_refit.shape[1]):
            if not theta_mask[i, j]:
                continue
            feat_tups.append((motif, mut_pos, j))

    for col_idx in range(theta.shape[1]):
        full_theta[:,col_idx], theta_lower[:,col_idx], theta_upper[:,col_idx] = \
                combine_thetas_and_get_conf_int(
                    feat_generator,
                    full_feat_generator,
                    theta_flat,
                    covariance_est,
                    col_idx,
                    best_model.z_stat,
                    feat_tups,
                )

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
