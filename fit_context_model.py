#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fit a context-sensitive motif model via MCMC-EM
"""

import sys
import argparse
import os
import os.path
import csv
import pickle
import logging as log
import time
import random
from multiprocessing import Pool

import numpy as np
import scipy.stats

from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator
from mutation_order_gibbs import MutationOrderGibbsSampler
from survival_problem_lasso import SurvivalProblemLasso
from likelihood_evaluator import LikelihoodComparer
from common import *
from read_data import *
from matsen_grp_data import *
from context_model_algo import ContextModelAlgo

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed',
        type=int,
        help='rng seed for replicability',
        default=1533)
    parser.add_argument('--input-genes',
        type=str,
        help='genes data in csv',
        default='_output/genes.csv')
    parser.add_argument('--input-seqs',
        type=str,
        help='sequence data in csv',
        default='_output/seqs.csv')
    parser.add_argument('--sample-regime',
        type=int,
        default=1,
        choices=(1, 2, 3),
        help='1: take all sequences; 2: sample random sequence from cluster; 3: choose most highly mutated sequence (default: 1)')
    parser.add_argument('--scratch-directory',
        type=str,
        help='where to write gibbs workers and dnapars files, if necessary',
        default='_output')
    parser.add_argument('--num-cpu-threads',
        type=int,
        help='number of threads to use during M-step',
        default=1)
    parser.add_argument('--num-jobs',
        type=int,
        help='number of jobs to submit during E-step',
        default=1)
    parser.add_argument('--motif-lens',
        type=str,
        help='length of motif (must be odd)',
        default='5')
    parser.add_argument('--positions-mutating',
        type=str,
        help="""
        length of left motif flank determining which position is mutating; comma-separated within
        a motif length, colon-separated between, e.g., --motif-lens 3,5 --left-flank-lens 0,1:0,1,2 will
        be a 3mer with first and second mutating position and 5mer with first, second and third
        """,
        default=None)
    parser.add_argument('--em-max-iters',
        type=int,
        help='number of EM iterations',
        default=20)
    parser.add_argument('--burn-in',
        type=int,
        help='number of burn-in iterations for E-step',
        default=10)
    parser.add_argument('--num-e-samples',
        type=int,
        help='number of base samples to draw during E-step',
        default=10)
    parser.add_argument('--log-file',
        type=str,
        help='log file',
        default='_output/context_log.txt')
    parser.add_argument('--out-file',
        type=str,
        help='file with pickled context model',
        default='_output/context_model.pkl')
    parser.add_argument('--theta-file',
        type=str,
        help='true theta file',
        default='')
    parser.add_argument("--penalty-params",
        type=str,
        help="penalty parameters, comma separated",
        default="0.5, 0.25")
    parser.add_argument('--tuning-sample-ratio',
        type=float,
        help="""
            proportion of data to use for tuning the penalty parameter.
            if zero, tunes by number of confidence intervals for theta that do not contain zero
            """,
        default=0.2)
    parser.add_argument('--num-val-burnin',
        type=int,
        help='Number of burn in iterations when estimating likelihood of validation data',
        default=10)
    parser.add_argument('--num-val-samples',
        type=int,
        help='Number of burn in iterations when estimating likelihood of validation data',
        default=10)
    parser.add_argument('--num-val-threads',
        type=int,
        help='number of threads to use for validation calculations',
        default=12)
    parser.add_argument('--validation-column',
        type=str,
        help='column in the dataset to split training/validation on (e.g., subject, clonal_family, etc.)',
        default=None)
    parser.add_argument('--per-target-model',
        action='store_true')
    parser.add_argument("--locus",
        type=str,
        choices=('','igh','igk','igl'),
        help="locus (igh, igk or igl; default empty)",
        default='')
    parser.add_argument("--species",
        type=str,
        choices=('','mouse','human'),
        help="species (mouse or human; default empty)",
        default='')
    parser.add_argument("--z-stat",
        type=float,
        help="confidence interval z statistic",
        default=1.96)

    parser.set_defaults(per_target_model=False, conf_int_stop=False)
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
    for m in args.motif_lens:
        assert(m % 2 == 1)

    args.max_motif_len = max(args.motif_lens)

    if args.positions_mutating is None:
        # default to central base mutating
        args.max_left_flank = None
        args.max_right_flank = None
    else:
        args.positions_mutating = [[int(m) for m in positions.split(',')] for positions in args.positions_mutating.split(':')]
        for motif_len, positions in zip(args.motif_lens, args.positions_mutating):
            for m in positions:
                assert(m in range(motif_len))

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
    return args

def main(args=sys.argv[1:]):
    args = parse_args()
    log.basicConfig(format="%(message)s", filename=args.log_file, level=log.DEBUG)
    np.random.seed(args.seed)

    true_theta = None
    if args.theta_file != "":
        true_theta, _ = load_true_model(args.theta_file)

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
        args.input_genes,
        args.input_seqs,
        motif_len=args.max_motif_len,
        left_flank_len=args.max_left_flank,
        right_flank_len=args.max_right_flank,
        sample=args.sample_regime,
        locus=args.locus,
        species=args.species,
    )

    train_idx, val_idx = split_train_val(
        len(obs_data),
        metadata,
        args.tuning_sample_ratio,
        args.validation_column,
    )
    train_set = [obs_data[i] for i in train_idx]
    val_set = [obs_data[i] for i in val_idx]
    feat_generator.add_base_features_for_list(train_set)
    feat_generator.add_base_features_for_list(val_set)

    log.info("Data statistics:")
    log.info("  Number of sequences: Train %d, Val %d" % (len(train_idx), len(val_idx)))
    log.info(get_data_statistics_print_lines(obs_data, feat_generator))
    log.info("Settings %s" % args)

    log.info("Running EM")
    cmodel_algo = ContextModelAlgo(feat_generator, obs_data, train_set, args, all_runs_pool)

    # Run EM on the lasso parameters from largest to smallest
    val_set_evaluator = None
    penalty_param_prev = None
    num_val_samples = args.num_val_samples
    results_list = []
    num_nonzero_confint = 0
    for penalty_param in args.penalty_params:
        log.info("==== Penalty parameter %f ====" % penalty_param)
        curr_model_results = cmodel_algo.fit(
            penalty_param,
            val_set_evaluator,
            reference_pen_param=penalty_param_prev
        )

        if args.tuning_sample_ratio > 0:
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

        if curr_model_results.penalized_num_nonzero > 0:
            # first make sure that the penalty isnt so big that theta is empty
            if args.tuning_sample_ratio > 0:
                # We are going to tune using EM surrogate function on the validation set
                ll_lower_bound = curr_model_results.log_lik_ratio_lower_bound
                if ll_lower_bound is not None and ll_lower_bound < 0:
                    # This model is not better than the previous model.
                    # Use a greedy approach and stop trying penalty parameters
                    log.info("EM surrogate function is decreasing. Stop trying penalty parameters")
                    break
            else:
                # We are going to tune using confidence intervals
                if curr_model_results.num_not_crossing_zero < num_nonzero_confint:
                    log.info("Number of nonzero confidence intervals decreasing. Stop trying penalty parameters")
                    break
        num_nonzero_confint = curr_model_results.num_not_crossing_zero
        penalty_param_prev = penalty_param

    if all_runs_pool is not None:
        all_runs_pool.close()
        # helpful comment copied over: make sure we don't keep these processes open!
        all_runs_pool.join()
    log.info("Completed!")

if __name__ == "__main__":
    main(sys.argv[1:])
