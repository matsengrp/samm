#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fit a context-sensitive motif model via MCMC-EM
Use bracketing to determine the penalty parameter that maximizes the number of
confidence intervals that don't cross zero
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
import scipy.optimize

from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator
from mutation_order_gibbs import MutationOrderGibbsSampler
from survival_problem_lasso import SurvivalProblemLasso
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
        default=1)
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
    parser.add_argument("--penalty-param-min",
        type=float,
        default=0.01)
    parser.add_argument("--penalty-param-max",
        type=float,
        default=10)
    parser.add_argument("--max-search",
        type=int,
        help="maximum number of penalty parameters to try",
        default=4)
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
    parser.add_argument('--z-stat',
        type=float,
        help="Determines the width of the confidence intervals",
        default=1.96)
    parser.add_argument('--nonzero-ratio',
        type=float,
        help="fraction of theta values with conf interval not crossing zero, if zero, just get the one with biggest number of nonzero crossings",
        default=0.9)

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

    args.intermediate_out_dir = os.path.dirname(args.out_file)

    args.scratch_dir = os.path.join(args.scratch_directory, str(time.time() + np.random.randint(10000)))
    if not os.path.exists(args.scratch_dir):
        os.makedirs(args.scratch_dir)

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
        args.input_genes,
        args.input_seqs,
        motif_len=args.max_motif_len,
        left_flank_len=args.max_left_flank,
        right_flank_len=args.max_right_flank,
        sample=args.sample_regime,
        locus=args.locus,
        species=args.species,
    )
    feat_generator.add_base_features_for_list(obs_data)

    log.info("Data statistics:")
    log.info("  Number of sequences: %d" % len(obs_data))
    log.info(get_data_statistics_print_lines(obs_data, feat_generator))
    log.info("Settings %s" % args)

    log.info("Running EM")

    cmodel_algo = ContextModelAlgo(feat_generator, obs_data, obs_data, args, all_runs_pool)
    model_history = []
    def min_func(log_pen_param):
        fitted_model = cmodel_algo.fit(np.power(10, log_pen_param))
        model_history.append(fitted_model)
        with open(args.out_file, "w") as f:
            pickle.dump(model_history, f)

        if args.nonzero_ratio == 0:
            # optimize for number not crossing zero
            func_val = -fitted_model.num_not_crossing_zero
        else:
            # optimize for percent not crossing zero and for the percent to be close to the requested value
            func_val = np.abs(fitted_model.percent_not_crossing_zero - args.nonzero_ratio)
        log.info("log_pen_param %f, func_val %f" % (log_pen_param, func_val))
        return func_val

    # Use bracketing to find the penalty parameter that maximizes the number of confidence intervals
    # that don't cross zero
    optim_res = scipy.optimize.minimize_scalar(
        min_func,
        bounds=(np.log10(args.penalty_param_min), np.log10(args.penalty_param_max)),
        method='bounded',
        options={"maxiter": args.max_search},
    )

    if all_runs_pool is not None:
        all_runs_pool.close()
        # helpful comment copied over: make sure we don't keep these processes open!
        all_runs_pool.join()
    log.info("Completed. Best penalty parameter %f, opt value %d" % (optim_res.x, -optim_res.fun))

if __name__ == "__main__":
    main(sys.argv[1:])
