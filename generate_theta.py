"""
Randomly generate a mutation model (the theta parameters) according to the specified
motif model structure (e.g. 3-mer, 3,5-mers, or even with offsets).

Currently this assumes that the aggregate model is a k-mer motif model where k is odd
and the center position mutates.
"""
import pickle
import sys
import argparse
import numpy as np
import csv
import re

from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator
from common import *

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed',
        type=int,
        help='Random number generator seed for replicability',
        default=1)
    parser.add_argument('--mutability',
        type=str,
        default='R/shmulate_params/mut_mouse.csv',
        help='Path to mutability model file - used for sampling distribution for theta')
    parser.add_argument('--substitution',
        type=str,
        default='R/shmulate_params/sub_mouse.csv',
        help='Path to substitution model file - used for sampling distribution for theta')
    parser.add_argument('--output-model',
        type=str,
        help='Pickle file to output with true theta parameters',
        default='_output/true_model.pkl')
    parser.add_argument('--motif-lens',
        type=str,
        help='Comma-separated list of motif lengths',
        default="5")
    parser.add_argument('--positions-mutating',
        type=str,
        help="""
        A colon-separated list of comma-separated lists indicating the positions that are mutating in the true model.
        The colons separate based on motif length. Each comma-separated list corresponds to the
        positions that mutate for the same motif length. The positions are indexed starting from zero.
        e.g., --motif-lens 3,5 --left-flank-lens 0,1:0,1,2 will be a 3mer with first and second mutating position
        and 5mer with first, second and third
        """,
        default=None)
    parser.add_argument('--effect-size',
        type=float,
        help='How much to scale sampling distribution for theta',
        default=1.0)
    parser.add_argument('--nonzero-ratio',
        type=float,
        help='Proportion of motifs to be nonzero',
        default=0.5)
    parser.add_argument('--per-target-model',
        action="store_true",
        help='Allow different hazard rates for different target nucleotides')
    parser.add_argument('--use-shmulate-as-truth',
        action="store_true",
        help='Use s5nf parameters as the truth (MK_S5F in shazam)')

    parser.set_defaults(per_target_model=False, use_shmulate_as_truth=False)
    args = parser.parse_args()

    args.motif_lens = [int(m) for m in args.motif_lens.split(",")]
    args.positions_mutating, args.max_mut_pos = process_mutating_positions(args.motif_lens, args.positions_mutating)
    args.num_cols = NUM_NUCLEOTIDES + 1 if args.per_target_model else 1

    return args

def _read_mutability_probability_params(args):
    """
    Read S5F parameters and use a shuffled version if requested
    Note: a shuffled version should prevent bias when comparing
    performance between shazam and our survival model since the
    current S5F parameters were estimated by counting and then
    aggregating inner 3-mers.
    """
    exp_theta = []
    with open(args.mutability, "rb") as mutability_f:
        mut_reader = csv.reader(mutability_f, delimiter=" ")
        mut_reader.next()
        for row in mut_reader:
            motif = row[0].lower()
            exp_theta.append(float(row[1]))

    exp_theta = np.array(exp_theta)
    exp_theta = exp_theta.reshape((exp_theta.size, 1))

    if not args.per_target_model:
        return exp_theta
    else:
        probability_matrix = []
        with open(args.substitution, "rb") as substitution_f:
            substitution_reader = csv.reader(substitution_f, delimiter=" ")
            substitution_reader.next()
            for row in substitution_reader:
                motif = row[0].lower()
                probability_matrix.append([float(row[i]) for i in range(1, 5)])
        probability_matrix = np.array(probability_matrix)
        full_exp_theta = np.hstack([exp_theta, probability_matrix])
        assert(full_exp_theta.shape[1] == 5)
        return full_exp_theta

def _make_theta_sampling_distribution(args):
    shmulate_theta = _read_mutability_probability_params(args)
    nonzero_theta_col0 = shmulate_theta[shmulate_theta[:,0] != 0, 0]
    shmulate_theta_col0 = np.log(nonzero_theta_col0)
    shmulate_theta_col0 -= np.median(shmulate_theta_col0) # center shmulate parameters

    prob_theta = []
    for row_idx in range(shmulate_theta.shape[0]):
        prob_row = shmulate_theta[row_idx, 1:]
        prob_theta.append(prob_row[prob_row != 0])
    prob_theta = np.log(np.array(prob_theta))
    return shmulate_theta_col0, prob_theta

def _generate_true_parameters(hier_feat_generator, args, theta_sampling_col0, theta_sampling_col_prob):
    """
    Make a sparse version if nonzero_ratio > 0
    """
    num_cols = NUM_NUCLEOTIDES + 1 if args.per_target_model else 1
    theta_shape = (hier_feat_generator.feature_vec_len, num_cols)

    theta_mask = get_possible_motifs_to_targets(
        hier_feat_generator.motif_list,
        theta_shape,
        hier_feat_generator.mutating_pos_list,
    )

    theta_param = np.random.choice(
        theta_sampling_col0,
        size=(hier_feat_generator.feature_vec_len, 1),
        replace=True,
    )
    # Zero parts of the first theta column
    indices_to_zero = np.random.choice(
        np.arange(theta_param.size),
        size=int((1 - args.nonzero_ratio) * theta_param.size),
        replace=False,
    )
    theta_param[indices_to_zero] = 0
    theta_param -= np.median(theta_param)

    if args.per_target_model:
        theta_col_prob_idx = np.random.choice(
            np.arange(theta_sampling_col_prob.shape[0]),
            size=hier_feat_generator.feature_vec_len,
            replace=True,
        )

        # zero out certain rows -- set to equal prob 1/3
        col_probs_to_third = np.random.choice(
            np.arange(hier_feat_generator.feature_vec_len),
            size=int((1 - args.nonzero_ratio) * hier_feat_generator.feature_vec_len),
            replace=False,
        )

        theta_col_probs = []
        for row_idx, sampled_row_idx in enumerate(theta_col_prob_idx):
            theta_row_mask = np.where(theta_mask[row_idx, 1:])[0]
            theta_col_prob_row = np.ones(NUM_NUCLEOTIDES) * -np.inf
            if row_idx in col_probs_to_third:
                theta_col_prob_row[theta_row_mask] = np.log(1.0/3)
            else:
                theta_col_prob_row[theta_row_mask] = theta_sampling_col_prob[sampled_row_idx]
            theta_col_probs.append(theta_col_prob_row)
        theta_col_probs = np.array(theta_col_probs)
        theta_param = np.hstack((theta_param, theta_col_probs))

    return theta_param, theta_mask

def dump_parameters(agg_theta, theta, args, feat_generator):
    # Dump a pickle file of simulation parameters
    with open(args.output_model, 'w') as f:
        pickle.dump((agg_theta, theta), f)

    # Dump a text file of theta for easy viewing
    with open(re.sub('.pkl', '.txt', args.output_model), 'w') as f:
        f.write("True Theta\n")
        lines = get_nonzero_theta_print_lines(theta, feat_generator)
        f.write(lines)

def main(args=sys.argv[1:]):
    args = parse_args()

    # Randomly generate number of mutations or use default
    np.random.seed(args.seed)

    hier_feat_generator = HierarchicalMotifFeatureGenerator(
        motif_lens=args.motif_lens,
        left_motif_flank_len_list=args.positions_mutating,
    )
    agg_feat_generator = HierarchicalMotifFeatureGenerator(
        motif_lens=[hier_feat_generator.max_motif_len],
        left_motif_flank_len_list=args.max_mut_pos,
    )

    if args.use_shmulate_as_truth:
        h5f_theta = _read_mutability_probability_params(args)
        h5f_theta[h5f_theta == 0] = -np.inf
        h5f_theta[h5f_theta != -np.inf] = np.log(h5f_theta[h5f_theta != -np.inf])
        agg_h5f_theta = h5f_theta[:, 0:1] + h5f_theta[:, 1:]
        dump_parameters(agg_h5f_theta, h5f_theta, args, hier_feat_generator)
    else:
        theta_sampling_col0, theta_sampling_col_prob = _make_theta_sampling_distribution(args)
        avg_sampled_magnitude = np.sqrt(np.var(theta_sampling_col0))
        theta_raw, theta_mask = _generate_true_parameters(
            hier_feat_generator,
            args,
            theta_sampling_col0,
            theta_sampling_col_prob,
        )

        agg_theta_raw = create_aggregate_theta(hier_feat_generator, agg_feat_generator, theta_raw, np.zeros(theta_raw.shape, dtype=bool), theta_mask, keep_col0=False)

        # Now rescale theta according to effect size
        mult_factor = 1.0/np.sqrt(np.var(agg_theta_raw[agg_theta_raw != -np.inf])) * args.effect_size * avg_sampled_magnitude
        agg_theta = agg_theta_raw * mult_factor
        theta_raw = theta_raw * mult_factor

        dump_parameters(agg_theta, theta_raw, args, hier_feat_generator)

if __name__ == "__main__":
    main(sys.argv[1:])
