"""
This can do offset motifs, but you need to have the center position of the longest motif mutate.
And we will suppose that all the other shorter motifs are contained in the longest motif.
"""
import pickle
import sys
import argparse
import itertools
import numpy as np
import os
import os.path
import csv
import re
import random

from survival_model_simulator import SurvivalModelSimulatorSingleColumn
from survival_model_simulator import SurvivalModelSimulatorMultiColumn
from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator
from common import *
from read_data import GERMLINE_PARAM_FILE

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed',
        type=int,
        help='rng seed for replicability',
        default=1533)
    parser.add_argument('--random-gene-len',
        type=int,
        help='Create random germline genes of this length. If zero, load true germline genes',
        default=24)
    parser.add_argument('--mutability',
        type=str,
        default='gctree/S5F/Mutability.csv',
        help='path to mutability model file')
    parser.add_argument('--substitution',
        type=str,
        default='gctree/S5F/Substitution.csv',
        help='path to substitution model file')
    parser.add_argument('--output-model',
        type=str,
        help='true theta pickle file',
        default='_output/true_model.pkl')
    parser.add_argument('--motif-lens',
        type=str,
        help='length of motifs, comma separated',
        default="5")
    parser.add_argument('--positions-mutating',
        type=str,
        help="""
        length of left motif flank determining which position is mutating; comma-separated within
        a motif length, colon-separated between, e.g., --motif-lens 3,5 --left-flank-lens 0,1:0,1,2 will
        be a 3mer with first and second mutating position and 5mer with first, second and third
        """,
        default=None)
    parser.add_argument('--effect-size',
        type=float,
        help='how much to scale sampling distribution for theta',
        default=1.0)
    parser.add_argument('--sparsity-ratio',
        type=float,
        help='Proportion of motifs to be nonzero',
        default=0.5)
    parser.add_argument('--per-target-model',
        action="store_true",
        help='Allow different hazard rates for different target nucleotides')
    parser.add_argument('--shuffle',
        action="store_true",
        help='Use a shuffled version of the S5F parameters')

    parser.set_defaults(per_target_model=False, shuffle=False)
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
    nonzero_theta_vals = shmulate_theta[shmulate_theta != 0]
    shmulate_theta_vals = np.log(nonzero_theta_vals)
    shmulate_theta_vals -= np.median(shmulate_theta_vals) # center shmulate parameters
    return args.effect_size * shmulate_theta_vals

def _generate_true_parameters(hier_feat_generator, args, theta_sampling_distribution):
    """
    Make a sparse version if sparsity_ratio > 0
    """
    num_cols = NUM_NUCLEOTIDES + 1 if args.per_target_model else 1
    theta_shape = (hier_feat_generator.feature_vec_len, num_cols)

    theta_param = np.random.choice(theta_sampling_distribution, size=theta_shape, replace=True)
    theta_mask = get_possible_motifs_to_targets(
        hier_feat_generator.motif_list,
        theta_shape,
        hier_feat_generator.mutating_pos_list,
    )
    theta_param[~theta_mask] = -np.inf
    possible_indices = np.where(theta_mask)[0]
    indices_to_zero = np.random.choice(
        possible_indices,
        size=int((1 - args.sparsity_ratio) * possible_indices.size),
        replace=False,
    )
    for i in indices_to_zero:
        theta_param[i] = 0
    theta_param -= np.median(theta_param)
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

    theta_sampling_distribution = _make_theta_sampling_distribution(args)

    theta, theta_mask = _generate_true_parameters(
        hier_feat_generator,
        args,
        theta_sampling_distribution,
    )

    agg_theta = create_aggregate_theta(hier_feat_generator, agg_feat_generator, theta, np.zeros(theta.shape, dtype=bool), theta_mask, keep_col0=False)
    dump_parameters(agg_theta, theta, args, hier_feat_generator)

if __name__ == "__main__":
    main(sys.argv[1:])
