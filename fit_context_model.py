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
from likelihood_evaluator import LikelihoodComparer, GreedyLikelihoodComparer
from method_results import MethodResults
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
        default='.1')
    parser.add_argument('--tuning-sample-ratio',
        type=float,
        help="""
            proportion of data to use for tuning the penalty parameter.
            if zero, tunes by number of confidence intervals for theta that do not contain zero
            """,
        default=0.1)
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
    parser.add_argument('--refit-on-training',
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
    parser.add_argument("--val-mouse-idx",
        type=int,
        help="index of validation mouse",
        default=None)
    parser.add_argument("--omit-hessian",
        action="store_true",
        help="do not calculate the hessian")

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

    args.penalty_params = [float(p) for p in args.penalty_params.split(",")]
    args.penalty_params = sorted(args.penalty_params, reverse=True)

    assert(args.tuning_sample_ratio > 0)

    return args

def write_sampled_data(input_shazam_seqs, input_shazam_genes, sampled_set):
    """
    Write data after sampling so shazam and samm fit to the same data
    """

    with open(input_shazam_seqs, 'w') as shazam_seq_file, open(input_shazam_genes, 'w') as shazam_gene_file:
                gene_writer = csv.DictWriter(shazam_gene_file, ['germline_name', 'germline_sequence'])
                gene_writer.writeheader()
                seq_writer = csv.DictWriter(shazam_seq_file, ['germline_name', 'sequence'])
                seq_writer.writeheader()
                for idx, obs_seq_mutation in enumerate(sampled_set):
                    gl_name = 'germline' + str(idx)
                    gene_writer.writerow({'germline_name': gl_name,
                        'germline_sequence': obs_seq_mutation.start_seq_with_flanks})
                    seq_writer.writerow({'germline_name': gl_name,
                        'sequence': obs_seq_mutation.end_seq_with_flanks})

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
        args.val_mouse_idx,
    )
    train_set = [obs_data[i] for i in train_idx]
    val_set = [obs_data[i] for i in val_idx]

    # for fitting shazam and later validating
    write_sampled_data(
            args.out_file.replace('.pkl', '_train_seqs.csv'),
            args.out_file.replace('.pkl', '_train_genes.csv'),
            train_set
    )

    write_sampled_data(
            args.out_file.replace('.pkl', '_val_seqs.csv'),
            args.out_file.replace('.pkl', '_val_genes.csv'),
            val_set
    )

    if args.refit_on_training:
        # Only use the training set in computing and validating thetas
        obs_data = train_set
        train_idx, val_idx = split_train_val(
            len(obs_data),
            metadata,
            args.tuning_sample_ratio,
            validation_column=None,
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

    cmodel_algo = ContextModelAlgo(feat_generator, obs_data, train_set, args, all_runs_pool)

    # Run EM on the lasso parameters from largest to smallest
    log.info("Running EM")
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

        ll_ratio = curr_model_results.log_lik_ratio
        if curr_model_results.penalized_num_nonzero > 0 and ll_ratio is not None and ll_ratio < -ZERO_THRES:
            # Make sure that the penalty isnt so big that theta is empty
            # This model is not better than the previous model. Stop trying penalty parameters.
            # Time to refit the model
            log.info("EM surrogate function is decreasing. Stop trying penalty parameters. ll_ratio %f" % ll_ratio)
            break
        else:
            best_model_idx = param_i

        penalty_params_prev = penalty_params
        prev_pen_theta = curr_model_results.penalized_theta

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
