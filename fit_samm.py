"""
Fit a motif model using SAMM
"""

import sys
import argparse
import os
import os.path
import pickle
import logging as log
import time
from multiprocessing import Pool

import numpy as np

from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator
from mutation_order_gibbs import MutationOrderGibbsSampler
from survival_problem_lasso import SurvivalProblemLasso
from likelihood_evaluator import LikelihoodComparer
from samm_worker import SammWorker
from parallel_worker import MultiprocessingManager
from context_model_algo import ContextModelAlgo
import data_split
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
    parser.add_argument('--sample-regime',
        type=int,
        default=1,
        choices=(1, 2, 3),
        help='1: take all sequences; 2: sample random sequence from cluster; 3: choose most highly mutated sequence (default: 1)')
    parser.add_argument('--scratch-directory',
        type=str,
        help='Directory for writing temporary files',
        default='_output')
    parser.add_argument('--num-cpu-threads',
        type=int,
        help='Number of threads to use',
        default=1)
    parser.add_argument('--num-jobs',
        type=int,
        help='Number of jobs to submit to a Slurm cluster during E-step. (If using only 1 job, it does not submit to a cluster.)',
        default=1)
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
    parser.add_argument('--unpenalized-em-max-iters',
        type=int,
        help='Maximum number of EM iterations during the fitting procedure for each penalty parameter',
        default=20)
    parser.add_argument('--em-max-iters',
        type=int,
        help='Maximum number of EM iterations during the fitting procedure for each penalty parameter',
        default=20)
    parser.add_argument('--max-m-iters',
        type=int,
        help='Maximum number of iterations for M step in EM',
        default=200)
    parser.add_argument('--burn-in',
        type=int,
        help='Number of burn-in iterations used on the first E-step of each penalty parameter',
        default=10)
    parser.add_argument('--num-e-samples',
        type=int,
        help='Number of mutation order samples to draw per observation during E-step',
        default=10)
    parser.add_argument('--sampling-rate',
        type=int,
        help='Number of gibbs sweep to perform to get one sample',
        default=1)
    parser.add_argument('--log-file',
        type=str,
        help='Log file',
        default='_output/context_log.txt')
    parser.add_argument('--out-file',
        type=str,
        help='Output file with pickled context model',
        default='_output/context_model.pkl')
    parser.add_argument("--penalty-params",
        type=str,
        help="Comma-separated list of penalty parameters",
        default="0.5, 0.25")
    parser.add_argument('--tuning-sample-ratio',
        type=float,
        help="Proportion of data to reserve for tuning the penalty parameter",
        default=0.1)
    parser.add_argument('--validation-col',
        type=str,
        default=None)
    parser.add_argument('--k-folds',
        type=int,
        help="""
        Number of folds for cross validation (for tuning the penalty parameter).
        If <= 1, then do training-validation split.
        Otherwise, then do k-fold CV.
        """,
        default=1)
    parser.add_argument('--num-val-burnin',
        type=int,
        help='Number of burn-in iterations when estimating likelihood of validation data',
        default=10)
    parser.add_argument('--num-val-samples',
        type=int,
        help='Number of mutation order samples drawn per observation when estimating likelihood of validation data',
        default=10)
    parser.add_argument('--per-target-model',
        action='store_true',
        help='Fit a model that allows for different hazard rates for different target nucleotides')
    parser.add_argument("--omit-hessian",
        action="store_true",
        help="Do not calculate uncertainty intervals (omit calculating the hessian matrix)")

    parser.set_defaults(per_target_model=False, conf_int_stop=False, omit_hessian=False)
    args = parser.parse_args()

    # Determine problem solver
    args.problem_solver_cls = SurvivalProblemLasso

    # Determine sampler
    args.sampler_cls = MutationOrderGibbsSampler
    if args.per_target_model:
        # First column is the median theta value and the remaining columns are the offset for that target nucleotide
        args.theta_num_col = NUM_NUCLEOTIDES + 1
    else:
        args.theta_num_col = 1

    process_motif_length_args(args)

    args.intermediate_out_dir = os.path.dirname(args.out_file)

    args.scratch_dir = os.path.join(args.scratch_directory, str(time.time() + np.random.randint(10000)))
    if not os.path.exists(args.scratch_dir):
        os.makedirs(args.scratch_dir)

    # sort penalty params from largest to smallest
    args.penalty_params = [float(p) for p in args.penalty_params.split(",")]
    args.penalty_params = sorted(args.penalty_params, reverse=True)

    if len(args.penalty_params) > 1:
        assert args.tuning_sample_ratio > 0 or args.k_folds > 1

    return args

def _check_same_support(model_results):
    """
    List of MethodResults
    """
    theta_support = model_results[0].model_masks.zeroed_thetas
    for res in model_results[1:]:
        if np.any(theta_support != res.model_masks.zeroed_thetas):
            return False
    return True

def main(args=sys.argv[1:]):
    args = parse_args()
    log.basicConfig(format="%(message)s", filename=args.log_file, level=log.DEBUG)
    np.random.seed(args.seed)

    if max(args.k_folds, args.num_cpu_threads) > 1:
        all_runs_pool = Pool(max(args.k_folds, args.num_cpu_threads))
    else:
        all_runs_pool = None

    feat_generator = HierarchicalMotifFeatureGenerator(
        motif_lens=args.motif_lens,
        left_motif_flank_len_list=args.positions_mutating,
    )
    cmodel_algo = ContextModelAlgo(feat_generator, args)

    log.info("Reading data")
    obs_data, metadata = read_gene_seq_csv_data(
        args.input_naive,
        args.input_mutated,
        motif_len=args.max_motif_len,
        left_flank_len=args.max_left_flank,
        right_flank_len=args.max_right_flank,
    )

    feat_generator.add_base_features_for_list(obs_data)
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
        data_folds.append((train_set, val_set))

    st_time = time.time()
    log.info("Data statistics:")
    log.info("  Number of sequences: Train %d, Val %d" % (len(train_idx), len(val_idx)))
    log.info(get_data_statistics_print_lines(obs_data, feat_generator))
    log.info("Settings %s" % args)
    log.info("Running EM")

    # Run EM on the lasso parameters from largest to smallest
    results_list = []
    val_set_evaluators = [None for _ in fold_indices]
    prev_pen_theta = None
    best_model_idx = 0
    for param_i, penalty_param in enumerate(args.penalty_params):
        param_results = []
        penalty_params_prev = None if param_i == 0 else args.penalty_params[param_i - 1]
        target_penalty_param = penalty_param if args.per_target_model else 0
        penalty_params = (penalty_param, target_penalty_param)
        log.info("==== Penalty parameters %f, %f ====" % penalty_params)
        workers = []
        for fold_idx, (train_set, val_set) in enumerate(data_folds):
            log.info("========= Fold %d ==============" % fold_idx)
            prev_pen_theta = results_list[param_i - 1][fold_idx].penalized_theta if param_i else None
            val_set_evaluator = val_set_evaluators[fold_idx]
            # Use the same number of order samples as previous validation set if possible
            prev_num_val_samples = val_set_evaluator.num_samples if val_set_evaluator is not None else args.num_val_samples
            samm_worker = SammWorker(
                fold_idx,
                cmodel_algo,
                train_set,
                penalty_params,
                args.em_max_iters,
                val_set_evaluator,
                prev_pen_theta,
                penalty_params_prev,
                val_set,
                prev_num_val_samples,
                args)
            workers.append(samm_worker)
        if args.k_folds > 1:
            # We will be using the MultiprocessingManager handle fitting theta for each fold (so python's multiprocessing lib)
            manager = MultiprocessingManager(all_runs_pool, workers, num_approx_batches=len(workers))
            results = manager.run()
        else:
            # We will be using the MultiprocessingManager parallelize computations within the M-step
            results = [w.run(all_runs_pool) for w in workers]

        param_results = [r[0] for r in results]
        results_list.append(param_results)
        val_set_evaluators = [r[1] for r in results]

        with open(args.out_file, "w") as f:
            pickle.dump(results_list, f)

        nonzeros = np.array([res.penalized_num_nonzero for res in param_results])
        log_lik_ratios = np.array([r.log_lik_ratio for r in param_results])
        log.info("Log lik ratios %s" % log_lik_ratios)
        if any(nonzeros) and param_i > 0:
            cv_interval = get_interval(log_lik_ratios, zscore=1)
            log.info("log lik interval (%f, %f)" % cv_interval)
            if cv_interval[0] < -ZERO_THRES:
                # Make sure that the penalty isnt so big that theta is empty
                # One std error below the mean for the log lik ratios surrogate is negative
                # Time to stop shrinking penalty param
                # This model is not better than the previous model. Stop trying penalty parameters.
                # Time to refit the model
                log.info("EM surrogate function is decreasing. Stop trying penalty parameters. ll_ratios %s" % log_lik_ratios)
                break

        best_model_idx = param_i

        if np.mean(nonzeros) == feat_generator.feature_vec_len:
            # Model is saturated so stop fitting new parameters
            log.info("Model is saturated with %d parameters. Stop fitting." % np.mean(nonzeros))
            break

    # Pick out the best model
    # Make sure we have hte same support. Otherwise we need to refit
    if _check_same_support(results_list[best_model_idx]):
        # Just use the first fold as template for doing the refitting unpenalized
        method_res = results_list[best_model_idx][0]
    else:
        log.info("Need to refit to get the same support")
        # If support is not the same, refit penalized on all the data and get that support
        # Just use the first fold as template for doing the refitting unpenalized
        method_res_template = results_list[best_model_idx][0]
        prev_pen_theta = results_list[best_model_idx - 1][0].penalized_theta if best_model_idx else None
        method_res = cmodel_algo.fit_penalized(
            obs_data,
            method_res_template.penalty_params,
            max_em_iters=args.em_max_iters,
            init_theta=prev_pen_theta,
            pool=all_runs_pool,
        )

    # Finally ready to refit as unpenalized model
    if args.num_cpu_threads > 1 and all_runs_pool is None:
        all_runs_pool = Pool(args.num_cpu_threads)
    cmodel_algo.refit_unpenalized(
        obs_data,
        model_result=method_res,
        max_em_iters=args.unpenalized_em_max_iters,
        get_hessian=not args.omit_hessian,
        pool=all_runs_pool,
    )

    # Pickle the refitted theta
    with open(args.out_file, "w") as f:
        pickle.dump(results_list, f)

    if not args.omit_hessian:
        full_feat_generator = HierarchicalMotifFeatureGenerator(
            motif_lens=[args.max_motif_len],
            left_motif_flank_len_list=args.max_mut_pos,
        )
        num_agg_cols = NUM_NUCLEOTIDES if args.per_target_model else 1
        agg_start_col = 1 if args.per_target_model else 0

        try:
            feat_generator_stage2 = HierarchicalMotifFeatureGenerator(
                motif_lens=args.motif_lens,
                model_truncation=method_res.model_masks,
                left_motif_flank_len_list=args.positions_mutating,
            )
            for col_idx in range(num_agg_cols):
                full_theta, theta_lower, theta_upper = feat_generator_stage2.combine_thetas_and_get_conf_int(
                    method_res.refit_theta,
                    variance_est=method_res.variance_est,
                    col_idx=col_idx + agg_start_col,
                    add_targets=args.per_target_model,
                )
        except ValueError as e:
            print(e)

            log.info("No fits had positive variance estimates")

    if all_runs_pool is not None:
        all_runs_pool.close()
        # helpful comment copied over: make sure we don't keep these processes open!
        all_runs_pool.join()
    log.info("Completed! Time: %s" % str(time.time() - st_time))

if __name__ == "__main__":
    main(sys.argv[1:])
