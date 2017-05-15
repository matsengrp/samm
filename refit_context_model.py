#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fit a context-sensitive un-penalized version motif model via MCMC-EM
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

import numpy as np
import scipy.stats

from models import ObservedSequenceMutations
from mcmc_em import MCMC_EM
from submotif_feature_generator import SubmotifFeatureGenerator
from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator
from mutation_order_gibbs import MutationOrderGibbsSampler
from survival_problem_lasso import SurvivalProblemLasso
from likelihood_evaluator import *
from method_results import MethodResults
from common import *
from read_data import *
from matsen_grp_data import *
from multiprocessing import Pool

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
    parser.add_argument("--zero-motifs",
        type=str,
        help="motifs with constant zero theta value, csv file",
        default='')

    parser.set_defaults(per_target_model=False)
    args = parser.parse_args()

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

    args.scratch_dir = os.path.join(args.scratch_directory, str(time.time()))
    if not os.path.exists(args.scratch_dir):
        os.makedirs(args.scratch_dir)

    return args

def load_true_model(file_name):
    with open(file_name, "rb") as f:
        real_params = pickle.load(f)
        true_theta = real_params[2] if len(real_params) > 2 else real_params[0]
        probability_matrix = real_params[1]
    return true_theta, probability_matrix

def initialize_theta(theta_shape, possible_theta_mask, zero_theta_mask):
    """
    Initialize theta
    @param possible_theta_mask: set the negative of this mask to negative infinity theta values
    @param zero_theta_mask: set the negative of this mask to negative infinity theta values
    """
    theta = np.random.randn(theta_shape[0], theta_shape[1]) * 1e-3
    # Set the impossible thetas to -inf
    theta[~possible_theta_mask] = -np.inf
    # Set particular thetas to zero upon request
    theta[zero_theta_mask] = 0
    return theta

def main(args=sys.argv[1:]):
    args = parse_args()
    log.basicConfig(format="%(message)s", filename=args.log_file, level=log.DEBUG)
    np.random.seed(args.seed)

    motifs_to_remove, pos_to_remove, target_pairs_to_remove = read_zero_motif_csv(args.zero_motifs, args.per_target_model)
    feat_generator = HierarchicalMotifFeatureGenerator(
        motif_lens=args.motif_lens,
        motifs_to_remove=motifs_to_remove,
        pos_to_remove=pos_to_remove,
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
    obs_data_train = []
    for obs_datum in obs_data:
        obs_data_train.append(feat_generator.create_base_features(obs_datum))

    log.info("Data statistics:")
    log.info("  Number of sequences: Full Train %d" % (len(obs_data_train)))
    log.info("Settings %s" % args)

    log.info("Running EM")

    motif_list = feat_generator.motif_list
    mutating_pos_list = feat_generator.mutating_pos_list

    # Run EM on the lasso parameters from largest to smallest
    theta_shape = (feat_generator.feature_vec_len, args.theta_num_col)
    possible_theta_mask = get_possible_motifs_to_targets(motif_list, theta_shape, mutating_pos_list)
    zero_theta_mask = get_zero_theta_mask(target_pairs_to_remove, feat_generator, theta_shape)

    true_theta = None
    if args.theta_file != "":
        true_theta, _ = load_true_model(args.theta_file)

    if args.num_cpu_threads > 1:
        all_runs_pool = Pool(args.num_cpu_threads)
    else:
        all_runs_pool = None

    em_algo = MCMC_EM(
        args.sampler_cls,
        args.problem_solver_cls,
        base_num_e_samples=args.num_e_samples,
        num_jobs=args.num_jobs,
        scratch_dir=args.scratch_dir,
        pool=all_runs_pool,
    )

    burn_in = args.burn_in
    results_list = []
    best_models = []
    init_theta = initialize_theta(theta_shape, possible_theta_mask, zero_theta_mask)

    refit_theta, variance_est, _ = em_algo.run(
        obs_data_train,
        feat_generator,
        init_theta,
        possible_theta_mask=possible_theta_mask,
        zero_theta_mask=zero_theta_mask,
        burn_in=burn_in,
        penalty_params=(0,), # now fit with no penalty
        max_em_iters=args.em_max_iters,
        intermed_file_prefix="%s/e_samples_refit_" % (args.intermediate_out_dir),
        get_hessian=True,
    )
    curr_model_results = MethodResults([0])
    curr_model_results.set_refit_theta(refit_theta, variance_est)

    # We save the final theta (potentially trained over all the data)
    with open(args.out_file, "w") as f:
        pickle.dump(curr_model_results, f)

    if all_runs_pool is not None:
        all_runs_pool.close()
        # helpful comment copied over: make sure we don't keep these processes open!
        all_runs_pool.join()

if __name__ == "__main__":
    main(sys.argv[1:])
