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

import numpy as np
import scipy.stats

from models import ObservedSequenceMutations
from mcmc_em import MCMC_EM
from submotif_feature_generator import SubmotifFeatureGenerator
from mutation_order_gibbs import MutationOrderGibbsSampler
from survival_problem_cvxpy import SurvivalProblemLassoCVXPY
from survival_problem_cvxpy import SurvivalProblemFusedLassoCVXPY
from survival_problem_lasso import SurvivalProblemLasso
from survival_problem_fused_lasso_prox import SurvivalProblemFusedLassoProximal
from likelihood_evaluator import LogLikelihoodEvaluator
from multinomial_solver import MultinomialSolver
from common import *
from read_data import *
from matsen_grp_data import *

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
    parser.add_argument('--solver',
        type=str,
        help='CL = cvxpy lasso, CFL = cvxpy fused lasso, L = gradient descent lasso, FL = fused lasso, PFL = fused lasso with prox solver',
        choices=["CL", "CFL", "L", "FL"],
        default="L")
    parser.add_argument('--motif-len',
        type=int,
        help='length of motif (must be odd)',
        default=5)
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
    parser.add_argument("--penalty-params",
        type=str,
        help="penalty parameters, comma separated",
        default="0.1,0.01,0.001")
    parser.add_argument('--tuning-sample-ratio',
        type=float,
        help='proportion of data to use for tuning the penalty parameter. if zero, doesnt tune',
        default=0.1)
    parser.add_argument('--num-val-burnin',
        type=int,
        help='Number of burn in iterations when estimating likelihood of validation data',
        default=1)
    parser.add_argument('--full-train',
        action='store_true',
        help='True = train on training data, then evaluate on validation data, then train on all the data, false = train on training data and evaluate on validation data')
    parser.add_argument('--per-target-model',
        action='store_true')

    parser.set_defaults(per_target_model=False, full_train=False)
    args = parser.parse_args()

    # Determine problem solver
    args.problem_solver_cls = SurvivalProblemLasso
    if args.solver == "CL":
        args.problem_solver_cls = SurvivalProblemLassoCVXPY
    elif args.solver == "CFL":
        if args.per_target_model:
            raise NotImplementedError()
        else:
            args.problem_solver_cls = SurvivalProblemFusedLassoCVXPY
    elif args.solver == "L":
        args.problem_solver_cls = SurvivalProblemLasso
    elif args.solver == "FL":
        if args.per_target_model:
            raise NotImplementedError()
        else:
            args.problem_solver_cls = SurvivalProblemFusedLassoProximal

    # Determine sampler
    args.sampler_cls = MutationOrderGibbsSampler
    if args.per_target_model:
        args.theta_num_col = NUM_NUCLEOTIDES
    else:
        args.theta_num_col = 1

    assert(args.motif_len % 2 == 1 and args.motif_len > 1)

    return args

def create_train_val_sets(obs_data, feat_generator, args):
    num_obs = len(obs_data)
    val_size = int(args.tuning_sample_ratio * num_obs)
    if args.tuning_sample_ratio > 0:
        val_size = max(val_size, 1)
    permuted_idx = np.random.permutation(num_obs)
    train_idx = permuted_idx[:num_obs - val_size]
    val_idx = permuted_idx[num_obs - val_size:]
    train_set = []
    for i in train_idx:
        train_set.append(
            feat_generator.create_base_features(obs_data[i])
        )

    val_set = []
    for i in val_idx:
        val_set.append(
            feat_generator.create_base_features(obs_data[i])
        )
    return train_set, val_set

def main(args=sys.argv[1:]):
    args = parse_args()
    log.basicConfig(format="%(message)s", filename=args.log_file, level=log.DEBUG)
    np.random.seed(args.seed)

    scratch_dir = os.path.join(args.scratch_directory, str(time.time()))
    if not os.path.exists(scratch_dir):
        os.makedirs(scratch_dir)

    feat_generator = SubmotifFeatureGenerator(motif_len=args.motif_len)

    log.info("Reading data")
    obs_data = read_gene_seq_csv_data(args.input_genes, args.input_seqs, motif_len=args.motif_len, sample=args.sample_regime)
    train_set, val_set = create_train_val_sets(obs_data, feat_generator, args)

    log.info("Number of sequences: Train %d, Val %d" % (len(train_set), len(val_set)))
    log.info("Settings %s" % args)

    log.info("Running EM")

    motif_list = feat_generator.get_motif_list()

    # Run EM on the lasso parameters from largest to smallest
    penalty_params = [float(l) for l in args.penalty_params.split(",")]
    results_list = []

    theta = np.zeros((feat_generator.feature_vec_len, args.theta_num_col))
    # Set the impossible thetas to -inf
    theta_mask = get_possible_motifs_to_targets(motif_list, theta.shape)
    theta[~theta_mask] = -np.inf

    val_set_evaluator = LogLikelihoodEvaluator(
        val_set,
        args.sampler_cls,
        feat_generator,
        num_jobs=args.num_jobs,
        scratch_dir=scratch_dir,
    )

    em_algo = MCMC_EM(
        train_set,
        val_set,
        feat_generator,
        args.sampler_cls,
        args.problem_solver_cls,
        theta_mask = theta_mask,
        base_num_e_samples=args.num_e_samples,
        num_jobs=args.num_jobs,
        num_threads=args.num_cpu_threads,
        scratch_dir=scratch_dir,
    )
    val_burn_in = args.num_val_burnin
    burn_in = args.burn_in
    prev_val_log_lik = -np.inf
    val_log_lik = None
    for penalty_param in sorted(penalty_params, reverse=True):
        log.info("Penalty parameter %f" % penalty_param)
        theta, _ = em_algo.run(
            theta=theta,
            burn_in=burn_in,
            penalty_param=penalty_param,
            max_em_iters=args.em_max_iters,
            train_and_val=False
        )

        # Get log likelihood on the validation set for tuning penalty parameter
        if args.tuning_sample_ratio > 0:
            log.info("Calculating validation log likelihood for penalty param %f" % penalty_param)
            val_log_lik = val_set_evaluator.get_log_lik(theta, burn_in=val_burn_in)
            log.info("Validation log likelihood %f" % val_log_lik)
            if args.full_train:
                theta, _ = em_algo.run(
                    theta=theta,
                    burn_in=burn_in,
                    penalty_param=penalty_param,
                    max_em_iters=args.em_max_iters,
                    train_and_val=True
                )
        burn_in = 0 # Only use burn in at the very beginning
        val_burn_in = 0

        # Get the probabilities of the target nucleotides
        fitted_prob_vector = None
        if not args.per_target_model:
            fitted_prob_vector = MultinomialSolver.solve(obs_data, feat_generator, theta)
            log.info("=== Fitted Probability Vector ===")
            log.info(get_nonzero_theta_print_lines(fitted_prob_vector, motif_list))

        # We save the final theta (potentially trained over all the data)
        results_list.append((penalty_param, theta, fitted_prob_vector, val_log_lik))
        with open(args.out_file, "w") as f:
            pickle.dump(results_list, f)

        log.info("==== FINAL theta, penalty param %f ====" % penalty_param)
        log.info(get_nonzero_theta_print_lines(theta, motif_list))

        if args.tuning_sample_ratio:
            # Decide what to do next - stop or keep searching penalty parameters?
            if val_log_lik <= prev_val_log_lik:
                # This penalty parameter is performing worse. We're done!
                log.info("Stop trying penalty parameters")
                break
            prev_val_log_lik = val_log_lik

if __name__ == "__main__":
    main(sys.argv[1:])
