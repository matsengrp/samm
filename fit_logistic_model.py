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
from fit_model_common import process_motif_length_args
from common import *
from read_data import *

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
        default="1")
    parser.add_argument('--min-C',
        type=float,
        help='Minimum C (inverse of penalty param) for logistic regression in sckitlearn',
        default=-1)
    parser.add_argument('--max-C',
        type=float,
        help='Maximum C for logistic regression in sckitlearn',
        default=1)
    parser.add_argument('--per-target-model',
        action='store_true',
        help='Fit a model that allows for different hazard rates for different target nucleotides')
    parser.add_argument('--log-file',
        type=str,
        help='Log file',
        default='_output/logistic.log')
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
    
    X = []
    y = []
    for obs in obs_data:
        X.append(obs.feat_matrix_start)
        y += obs.mutated_indicator

    stacked_X = scipy.sparse.vstack(X)
    stacked_y = np.array(y)

    logistic_reg = LogisticRegressionCV(
            Cs=np.power(10, np.arange(args.min_C, args.max_C,0.1)),
            cv=3,
            penalty='l1',
            solver='liblinear',
            scoring="neg_log_loss",
            max_iter=10000,
            fit_intercept=False)
    logistic_reg.fit(stacked_X, stacked_y)
    log.info(logistic_reg.coefs_paths_)
    log.info("Best scores %s", logistic_reg.scores_)
    log.info("Best C %s", logistic_reg.C_)
    lines = get_nonzero_theta_print_lines(logistic_reg.coef_.T, feat_generator)
    log.info(lines)

if __name__ == "__main__":
    main(sys.argv[1:])
