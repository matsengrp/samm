"""
We fit a logistic regression model to understand mutation rates instead.
"""
import copy
import sys
import argparse
import os
import os.path
import logging as log
from sklearn.linear_model import LogisticRegressionCV
import scipy.sparse
import numpy as np

import data_split
from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator
from logistic_problem_cvxpy import LogisticRegressionMotif
from fit_model_common import process_motif_length_args
from model_truncation import ModelTruncation
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
    parser.add_argument('--penalty-params',
        type=str,
        help="Comma-separated list of penalty parameters",
        default="8.0, 4.0, 2.0, 1.0, 0.5, 0.25")
    parser.add_argument('--tuning-sample-ratio',
        type=float,
        help="Proportion of data to reserve for tuning the penalty parameter",
        default=0.1)
    parser.add_argument('--k-folds',
        type=int,
        help="num folds for CV",
        default=1)
    parser.add_argument('--validation-col',
        type=str,
        default=None)
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
    args.penalty_params = [float(p) for p in args.penalty_params.split(",")]
    args.penalty_params = sorted(args.penalty_params, reverse=True)

    return args

def get_X_y_matrices(obs_data, per_target_model):
    # Process data
    X = []
    ys = []
    for i, obs in enumerate(obs_data):
        X.append(obs.feat_matrix_start)
        if per_target_model:
            y_vec = np.zeros(obs.seq_len)
            for k, v in obs.mutation_pos_dict.iteritems():
                y_vec[k] = NUCLEOTIDE_DICT[v] + 1
            ys.append(y_vec)
        else:
            ys.append(np.array(obs.mutated_indicator))

    stacked_X = scipy.sparse.vstack(X).todense()
    stacked_y = np.concatenate(ys)
    return stacked_X, stacked_y

def get_best_penalty_param(penalty_params, data_folds, max_iters=2000):
    if len(penalty_params) == 1:
        return penalty_params[0]

    # Fit the models for each penalty parameter
    tot_validation_values = []
    for penalty_param in penalty_params:
        tot_validation_value = 0
        for fold_idx, (val_X, val_y, logistic_reg) in enumerate(data_folds):
            theta_raw, theta_part_agg, train_value = logistic_reg.solve(
                    lam_val=penalty_param,
                    max_iters=max_iters,
                    verbose=True)
            tot_validation_value += logistic_reg.score(val_X, val_y)
            log.info("theta support %d", np.sum(np.abs(theta_raw) > 1e-5))
        tot_validation_values.append([penalty_param, tot_validation_value])
        log.info("penalty_param %f, tot_val %f", penalty_param, tot_validation_value)

    tot_validation_values = np.array(tot_validation_values)
    best_pen_param = tot_validation_values[np.argmax(tot_validation_values[:,1]),0]
    return best_pen_param

def fit_to_data(obs_data, pen_param, theta_shape, per_target_model, max_iters=5000):
    obs_X, obs_y = get_X_y_matrices(obs_data, per_target_model)
    logistic_reg = LogisticRegressionMotif(
            theta_shape,
            obs_X,
            obs_y,
            per_target_model=per_target_model)
    _, theta, _ = logistic_reg.solve(pen_param,
                    max_iters=max_iters,
                    verbose=True)
    return theta

def main(args=sys.argv[1:]):
    args = parse_args()
    log.basicConfig(format="%(message)s", filename=args.log_file, level=log.DEBUG)
    np.random.seed(args.seed)

    feat_generator = HierarchicalMotifFeatureGenerator(
        motif_lens=args.motif_lens,
        left_motif_flank_len_list=args.positions_mutating,
    )
    theta_shape = (feat_generator.feature_vec_len, NUM_NUCLEOTIDES + 1 if args.per_target_model else 1)

    log.info("Reading data")
    obs_data, metadata = read_gene_seq_csv_data(
        args.input_naive,
        args.input_mutated,
        motif_len=args.max_motif_len,
        left_flank_len=args.max_left_flank,
        right_flank_len=args.max_right_flank,
    )
    log.info("num observations %d", len(obs_data))

    feat_generator.add_base_features_for_list(obs_data)

    # Process data
    fold_indices = data_split.split(
        len(obs_data),
        metadata,
        args.tuning_sample_ratio,
        args.k_folds,
        validation_column=args.validation_col,
    )
    data_folds = []
    for train_idx, val_idx in fold_indices:
        train_set = [obs_data[i] for i in train_idx]
        val_set = [obs_data[i] for i in val_idx]
        train_X, train_y = get_X_y_matrices(train_set, args.per_target_model)
        val_X, val_y = get_X_y_matrices(val_set, args.per_target_model)
        logistic_reg = LogisticRegressionMotif(
                theta_shape,
                train_X,
                train_y,
                per_target_model=args.per_target_model)
        data_folds.append((val_X, val_y, logistic_reg))

    # Fit the models for each penalty parameter
    best_pen_param = get_best_penalty_param(args.penalty_params, data_folds)
    log.info("best penalty param %f", best_pen_param)

    # Refit penalized with all the data
    penalized_theta = fit_to_data(
            obs_data,
            best_pen_param,
            theta_shape,
            args.per_target_model)
    model_masks = ModelTruncation(penalized_theta, feat_generator)

    # Create obs data with new pruned feature generator
    feat_generator_stage2 = copy.deepcopy(feat_generator)
    feat_generator_stage2.update_feats_after_removing(model_masks)
    # Refit with all the data -- no penalty!
    obs_data_stage2 = [copy.deepcopy(o) for o in obs_data]
    feat_generator_stage2.add_base_features_for_list(obs_data_stage2)
    refit_theta_shape = (feat_generator_stage2.feature_vec_len, NUM_NUCLEOTIDES + 1 if args.per_target_model else 1)
    refit_theta = fit_to_data(
            obs_data_stage2,
            0,
            refit_theta_shape,
            args.per_target_model)

    # Aggregate theta
    theta_agg = feat_generator_stage2.create_aggregate_theta(refit_theta)

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
