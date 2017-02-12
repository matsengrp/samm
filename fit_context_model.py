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

import numpy as np
import scipy.stats

from models import ObservedSequenceMutations
from mcmc_em import MCMC_EM
from feature_generator import SubmotifFeatureGenerator
from feat_generator_sparse import MotifFeatureGenerator
from mutation_order_gibbs import MutationOrderGibbsSampler
from mutation_order_gibbs import MutationOrderGibbsSamplerMultiTarget
from survival_problem_cvxpy import SurvivalProblemLassoCVXPY
from survival_problem_cvxpy import SurvivalProblemFusedLassoCVXPY
from survival_problem_lasso import SurvivalProblemLasso
from survival_problem_fused_lasso_prox import SurvivalProblemFusedLassoProximal
from common import *

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed',
        type=int,
        help='rng seed for replicability',
        default=1533)
    parser.add_argument('--input-file',
        type=str,
        help='sequence data in csv',
        default='_output/seqs.csv')
    parser.add_argument('--input-genes',
        type=str,
        help='genes data in csv',
        default='_output/genes.csv')
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
        help='CL = cvxpy lasso, CFL = cvxpy fused lasso, L = gradient descent lasso, FL = fused lasso with prox solver',
        choices=["CL", "CFL", "L", "FL"],
        default="FL")
    parser.add_argument('--motif-len',
        type=int,
        help='length of motif (must be odd)',
        default=3)
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
    parser.add_argument("--penalty-params",
        type=str,
        help="penalty parameters, comma separated",
        default="0.001")
    parser.add_argument('--theta-file',
        type=str,
        help='file with pickled true context model',
        default='_output/true_theta.pkl')
    parser.add_argument('--per-target-model',
        action='store_true')

    parser.set_defaults(per_target_model=False)
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
    if args.per_target_model:
        args.sampler_cls = MutationOrderGibbsSamplerMultiTarget
        args.theta_num_col = NUM_NUCLEOTIDES
    else:
        args.sampler_cls = MutationOrderGibbsSampler
        args.theta_num_col = 1

    assert(args.motif_len % 2 == 1 and args.motif_len > 1)

    return args

def load_true_theta(theta_file, per_target_model):
    """
    @param theta_file: file name
    @param per_target_model: if True, we return the entire theta. If False, we return a collapsed theta vector

    @return the true theta vector/matrix
    """
    true_theta = pickle.load(open(theta_file, 'rb'))
    if per_target_model:
        return true_theta
    else:
        return np.matrix(np.max(true_theta, axis=1)).T

def main(args=sys.argv[1:]):
    args = parse_args()
    log.basicConfig(format="%(message)s", filename=args.log_file, level=log.DEBUG)
    np.random.seed(args.seed)
    # feat_generator = SubmotifFeatureGenerator(motif_len=args.motif_len)
    feat_generator = MotifFeatureGenerator(motif_len=args.motif_len)

    # Load true theta for comparison
    true_theta = load_true_theta(args.theta_file, args.per_target_model)
    assert(true_theta.shape[0] == feat_generator.feature_vec_len)

    log.info("Reading data")
    gene_dict, obs_data = read_gene_seq_csv_data(args.input_genes, args.input_file)
    log.info("Number of sequences %d" % len(obs_data))
    log.info("Settings %s" % args)

    log.info("Running EM")

    motif_list = feat_generator.get_motif_list()

    # Run EM on the lasso parameters from largest to smallest
    penalty_params = [float(l) for l in args.penalty_params.split(",")]
    results_list = []

    theta = np.random.randn(feat_generator.feature_vec_len, args.theta_num_col)
    # Set the impossible thetas to -inf
    theta_mask = get_possible_motifs_to_targets(motif_list, theta.shape)
    theta[~theta_mask] = -np.inf

    em_algo = MCMC_EM(
        obs_data,
        feat_generator,
        args.sampler_cls,
        args.problem_solver_cls,
        theta_mask = theta_mask,
        base_num_e_samples=args.num_e_samples,
        burn_in=args.burn_in,
        num_jobs=args.num_jobs,
        num_threads=args.num_cpu_threads,
        approx='none',
    )

    for penalty_param in sorted(penalty_params, reverse=True):
        log.info("Penalty parameter %f" % penalty_param)
        theta, _ = em_algo.run(
            theta=theta,
            penalty_param=penalty_param,
            max_em_iters=args.em_max_iters,
        )
        results_list.append((penalty_param, theta))

        log.info("==== FINAL theta, penalty param %f ====" % penalty_param)
        log.info(get_nonzero_theta_print_lines(theta, motif_list))

        theta_shape = (theta_mask.sum(), 1)
        flat_theta = theta[theta_mask].reshape(theta_shape)
        flat_true_theta = true_theta[theta_mask].reshape(theta_shape)
        log.info("Spearman cor=%f, p=%f" % scipy.stats.spearmanr(flat_theta, flat_true_theta))
        log.info("Kendall Tau cor=%f, p=%f" % scipy.stats.kendalltau(flat_theta, flat_true_theta))
        log.info("Pearson cor=%f, p=%f" % scipy.stats.pearsonr(flat_theta, flat_true_theta))
        log.info("L2 error %f" % np.linalg.norm(flat_theta - flat_true_theta))

        with open(args.out_file, "w") as f:
            pickle.dump(results_list, f)

if __name__ == "__main__":
    main(sys.argv[1:])
