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
from context_model_algo import ContextModelAlgo
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
    parser.add_argument('--em-max-iters',
        type=int,
        help='Maximum number of EM iterations during the fitting procedure for each penalty parameter',
        default=20)
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

    args.motif_lens = [int(m) for m in args.motif_lens.split(',')]

    args.max_motif_len = max(args.motif_lens)

    args.positions_mutating, args.max_mut_pos = process_mutating_positions(args.motif_lens, args.positions_mutating)
    # Find the maximum left and right flanks of the motif with the largest length in the
    # hierarchy in order to process the data correctly
    args.max_left_flank = max(sum(args.positions_mutating, []))
    args.max_right_flank = max([motif_len - 1 - min(left_flanks) for motif_len, left_flanks in zip(args.motif_lens, args.positions_mutating)])

    # Check if our full feature generator will conform to input
    max_left_flanks = args.positions_mutating[args.motif_lens.index(args.max_motif_len)]
    if args.max_left_flank > max(max_left_flanks) or args.max_right_flank > args.max_motif_len - min(max_left_flanks) - 1:
        raise AssertionError('The maximum length motif does not contain all smaller length motifs.')

    args.intermediate_out_dir = os.path.dirname(args.out_file)

    args.scratch_dir = os.path.join(args.scratch_directory, str(time.time() + np.random.randint(10000)))
    if not os.path.exists(args.scratch_dir):
        os.makedirs(args.scratch_dir)

    # sort penalty params from largest to smallest
    args.penalty_params = [float(p) for p in args.penalty_params.split(",")]
    args.penalty_params = sorted(args.penalty_params, reverse=True)

    assert(args.tuning_sample_ratio > 0)

    return args

def main(args=sys.argv[1:]):
    args = parse_args()
    log.basicConfig(format="%(message)s", filename=args.log_file, level=log.DEBUG)
    np.random.seed(args.seed)

    if args.num_cpu_threads > 1:
        all_runs_pool = Pool(args.num_cpu_threads)
    else:
        all_runs_pool = None

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

    train_idx, val_idx = split_train_val(
        len(obs_data),
        metadata,
        args.tuning_sample_ratio,
        validation_column=args.validation_col,
    )
    train_set = [obs_data[i] for i in train_idx]
    val_set = [obs_data[i] for i in val_idx]
    feat_generator.add_base_features_for_list(train_set)
    feat_generator.add_base_features_for_list(val_set)

    st_time = time.time()
    log.info("Data statistics:")
    log.info("  Number of sequences: Train %d, Val %d" % (len(train_idx), len(val_idx)))
    log.info(get_data_statistics_print_lines(obs_data, feat_generator))
    log.info("Settings %s" % args)
    log.info("Running EM")
    cmodel_algo = ContextModelAlgo(feat_generator, obs_data, train_set, args, all_runs_pool)

    # Run EM on the lasso parameters from largest to smallest
    num_val_samples = args.num_val_samples
    results_list = []
    val_set_evaluator = None
    penalty_params_prev = None
    prev_pen_theta = None
    best_model_idx = 0
    for param_i, penalty_param in enumerate(args.penalty_params):
        target_penalty_param = penalty_param if args.per_target_model else 0
        penalty_params = (penalty_param, target_penalty_param)
        log.info("==== Penalty parameters %f, %f ====" % penalty_params)
        curr_model_results = cmodel_algo.fit_penalized(
            penalty_params,
            max_em_iters=args.em_max_iters,
            val_set_evaluator=val_set_evaluator,
            init_theta=prev_pen_theta,
            reference_pen_param=penalty_params_prev,
        )

        # Create this val set evaluator for next time
        val_set_evaluator = LikelihoodComparer(
            val_set,
            feat_generator,
            theta_ref=curr_model_results.penalized_theta,
            num_samples=num_val_samples,
            burn_in=args.num_val_burnin,
            num_jobs=args.num_jobs,
            scratch_dir=args.scratch_dir,
            pool=all_runs_pool,
        )
        # grab this many validation samples from now on
        num_val_samples = val_set_evaluator.num_samples

        # Save model results
        results_list.append(curr_model_results)
        with open(args.out_file, "w") as f:
            pickle.dump(results_list, f)

        ll_ratio = curr_model_results.log_lik_ratio_lower_bound
        if curr_model_results.penalized_num_nonzero > 0 and ll_ratio is not None and ll_ratio < -ZERO_THRES:
            # Make sure that the penalty isnt so big that theta is empty
            # This model is not better than the previous model. Stop trying penalty parameters.
            # Time to refit the model
            log.info("EM surrogate function is decreasing. Stop trying penalty parameters. ll_ratio %f" % ll_ratio)
            break
        else:
            best_model_idx = param_i

        if curr_model_results.penalized_num_nonzero == feat_generator.feature_vec_len:
            # Model is saturated so stop fitting new parameters
            log.info("Model is saturated with %d parameters. Stop fitting." % curr_model_results.penalized_num_nonzero)
            break

        penalty_params_prev = penalty_params
        prev_pen_theta = curr_model_results.penalized_theta

    cmodel_algo.refit_unpenalized(
        model_result=results_list[best_model_idx],
        max_em_iters=args.em_max_iters * 3,
        get_hessian=not args.omit_hessian,
    )

    # Pickle the refitted theta
    with open(args.out_file, "w") as f:
        pickle.dump(results_list, f)

    if not args.omit_hessian:
        full_feat_generator = HierarchicalMotifFeatureGenerator(
            motif_lens=[args.max_motif_len],
            left_motif_flank_len_list=args.max_mut_pos,
        )
        method_res = results_list[best_model_idx]
        num_agg_cols = NUM_NUCLEOTIDES if args.per_target_model else 1
        agg_start_col = 1 if args.per_target_model else 0

        try:
            feat_generator_stage2 = HierarchicalMotifFeatureGenerator(
                motif_lens=args.motif_lens,
                feats_to_remove=method_res.model_masks.feats_to_remove,
                left_motif_flank_len_list=args.positions_mutating,
            )
            for col_idx in range(num_agg_cols):
                full_theta, theta_lower, theta_upper = combine_thetas_and_get_conf_int(
                    feat_generator_stage2,
                    full_feat_generator,
                    method_res.refit_theta,
                    method_res.model_masks.zero_theta_mask_refit,
                    method_res.refit_possible_theta_mask,
                    method_res.sample_obs_info,
                    col_idx + agg_start_col,
                )
        except ValueError as e:
            print(e)

            log.info("Variance estimates negative; trying previous penalty parameter")
            if best_model_idx == 0:
                log.info("No fits had positive variance estimates")
            else:
                best_model_idx -= 1
                cmodel_algo.refit_unpenalized(
                    model_result=results_list[best_model_idx],
                    max_em_iters=args.em_max_iters,
                    get_hessian=not args.omit_hessian,
                )

        # Pickle the refitted theta
        with open(args.out_file, "w") as f:
            pickle.dump(results_list, f)

    if all_runs_pool is not None:
        all_runs_pool.close()
        # helpful comment copied over: make sure we don't keep these processes open!
        all_runs_pool.join()
    log.info("Completed! Time: %s" % str(time.time() - st_time))

if __name__ == "__main__":
    main(sys.argv[1:])
