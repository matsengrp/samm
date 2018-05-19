"""
We fit a logistic regression model to understand mutation rates instead.
"""

import sys
import argparse
import os
import os.path
import logging as log
from sklearn.linear_model import LogisticRegressionCV
import scipy.sparse

import numpy as np

from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator
from logistic_problem_cvxpy import LogisticRegressionMotif
from fit_model_common import process_motif_length_args
from common import *
from read_data import *

class LogisticModel:
    def __init__(self, theta):
        self.agg_refit_theta = theta

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed',
        type=int,
        help='Random number generator seed for replicability',
        default=1)
    parser.add_argument('--input-naive',
        type=str,
        help='Input CSV file with naive sequences',
        default='_output/naive.csv')
    parser.add_argument('--input-mutated',
        type=str,
        help='Input CSV file with naive sequences',
        default='_output/mutated.csv')
    parser.add_argument('--motif-lens',
        type=str,
        help='Comma-separated list of motif lengths for the motif model we are fitting',
        default='3')
    parser.add_argument('--positions-mutating',
        type=str,
        help="""
        A colon-separated list of comma-separated lists indicating the positions that are mutating in the motif model
        we are fitting. The colons separate based on motif length. Each comma-separated list corresponds to the
        positions that mutate for the same motif length. The positions are indexed starting from zero.
        e.g., --motif-lens 3,5 --left-flank-lens 0,1:0,1,2 will be a 3mer with first and second mutating position
        and 5mer with first, second and third
        """,
        default=None)
    parser.add_argument('--min-C',
        type=float,
        help='Minimum C (inverse of penalty param) for logistic regression in sckitlearn',
        default=0)
    parser.add_argument('--max-C',
        type=float,
        help='Maximum C for logistic regression in sckitlearn',
        default=0.4)
    parser.add_argument('--per-target-model',
        action='store_true',
        help='Fit a model that allows for different hazard rates for different target nucleotides')
    parser.add_argument('--log-file',
        type=str,
        help='Log file',
        default='_output/logistic.log')
    parser.add_argument('--model-pkl',
        type=str,
        help='pickled fitted model file',
        default='_output/logistic.pkl')
    parser.set_defaults(per_target_model=False)
    args = parser.parse_args()

    if args.per_target_model:
        # First column is the median theta value and the remaining columns are the offset for that target nucleotide
        args.theta_num_col = NUM_NUCLEOTIDES + 1
    else:
        args.theta_num_col = 1

    process_motif_length_args(args)

    return args

def main(args=sys.argv[1:]):
    args = parse_args()
    log.basicConfig(format="%(message)s", filename=args.log_file, level=log.DEBUG)
    np.random.seed(args.seed)

    feat_generator = HierarchicalMotifFeatureGenerator(
        motif_lens=args.motif_lens,
        left_motif_flank_len_list=args.positions_mutating,
    )

    log.info("Reading data")
    obs_data, metadata = read_gene_seq_csv_data(
        args.input_naive,
        args.input_mutated,
        motif_len=args.max_motif_len,
        left_flank_len=args.max_left_flank,
        right_flank_len=args.max_right_flank,
    )

    feat_generator.add_base_features_for_list(obs_data)
    
    # Process data
    X = []
    ys = []
    for i, obs in enumerate(obs_data):
        X.append(obs.feat_matrix_start)
        if args.per_target_model:
            y_vec = np.zeros(obs.seq_len)
            for k, v in obs.mutation_pos_dict.iteritems():
                y_vec[k] = NUCLEOTIDE_DICT[v] + 1
            ys.append(y_vec)
        else:
            ys.append(np.array(obs.mutated_indicator))

    stacked_X = scipy.sparse.vstack(X).todense()
    stacked_y = np.concatenate(ys)

    if args.per_target_model:
        theta_shape = (stacked_X.shape[1], NUM_NUCLEOTIDES + 1)
    else:
        theta_shape = (stacked_X.shape[1], 1)
    # Fit the model
    logistic_reg = LogisticRegressionMotif(
            theta_shape,
            stacked_X,
            stacked_y)
    theta, val = logistic_reg.solve(max_iters=2000)

    # Aggregate theta
    full_feat_generator = MotifFeatureGenerator(
        motif_len=args.max_motif_len,
        distance_to_start_of_motif=-args.max_left_flank,
    )
    hier_full_feat_generator = HierarchicalMotifFeatureGenerator(
        motif_lens=[args.max_motif_len],
        left_motif_flank_len_list=[[args.max_left_flank]],
    )
    motif_seq_mutations = []
    for motif in full_feat_generator.motif_list:
        obs_seq_mut = ObservedSequenceMutations(
            motif,
            mutate_string(motif, args.max_left_flank, "b"),
            motif_len=args.max_motif_len)
        motif_seq_mutations.append(obs_seq_mut)

    feat_generator.add_base_features_for_list(motif_seq_mutations)

    agg_X = []
    for motif_idx, motif_obs in enumerate(motif_seq_mutations):
        agg_X.append(motif_obs.feat_matrix_start)
    agg_X = scipy.sparse.vstack(agg_X)

    theta_agg = agg_X * theta

    # Convert theta to log probability of mutation
    if args.per_target_model:
        theta_est = -np.log(1.0 + np.exp(-theta_agg))
        possible_mask = hier_full_feat_generator.get_possible_motifs_to_targets(theta_est.shape)
        theta_est[~possible_mask] = -np.inf
    else:
        theta_est = np.log(1.0/(1.0 + np.exp(-theta_agg)))
    lines = get_nonzero_theta_print_lines(theta_est, feat_generator)
    log.info(lines)

    with open(args.model_pkl, "w") as f:
        pickle.dump(LogisticModel(theta_est), f)

if __name__ == "__main__":
    main(sys.argv[1:])
