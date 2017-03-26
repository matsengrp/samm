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
from likelihood_evaluator import *
from multinomial_solver import MultinomialSolver
from method_results import MethodResults
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
    parser.add_argument('--solver',
        type=str,
        help='CL = cvxpy lasso, CFL = cvxpy fused lasso, L = gradient descent lasso, FL = fused lasso, SFL = sparse fused lasso,',
        choices=["CL", "CFL", "L", "FL", "SFL"],
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
    parser.add_argument('--theta-file',
        type=str,
        help='true theta file',
        default='')
    parser.add_argument("--penalty-params",
        type=str,
        help="penalty parameters, comma separated",
        default="0.1, 0.01, 0.001")
    parser.add_argument("--fuse-center",
        type=str,
        help="center motif lengths, comma separated",
        default="")
    parser.add_argument('--tuning-sample-ratio',
        type=float,
        help='proportion of data to use for tuning the penalty parameter. if zero, doesnt tune',
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
    parser.add_argument('--chibs',
        action='store_true',
        help='True = estimate the marginal likelihood via Chibs')
    parser.add_argument('--validation-column',
        type=str,
        help='column in the dataset to split training/validation on (e.g., subject, clonal_family, etc.)',
        default=None)
    parser.add_argument('--full-train',
        action='store_true',
        help='True = train on training data, then evaluate on validation data, then train on all the data, false = train on training data and evaluate on validation data')
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

    parser.set_defaults(per_target_model=False, full_train=False, chibs=False)
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
    elif args.solver == "FL" or args.solver == "SFL":
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

    if args.fuse_center and args.problem_solver_cls != SurvivalProblemLasso:
        args.fuse_center = [int(k) for k in args.fuse_center.split(",")]
        for k in args.fuse_center:
            assert(k % 2 == 1) # all center fusions must be odd length
    else:
        args.fuse_center = []

    args.intermediate_out_file = args.out_file.replace(".pkl", "_intermed.pkl")

    args.scratch_dir = os.path.join(args.scratch_directory, str(time.time()))
    if not os.path.exists(args.scratch_dir):
        os.makedirs(args.scratch_dir)

    return args

def load_true_model(file_name):
    with open(file_name, "rb") as f:
        true_theta, probability_matrix = pickle.load(f)
        return true_theta, probability_matrix

def create_train_val_sets(obs_data, feat_generator, metadata, tuning_sample_ratio, validation_column):
    """
    @param obs_data: observed mutation data
    @param feat_generator: submotif feature generator
    @param metadata: metadata to include variables to perform validation on
    @param tuning_sample_ratio: ratio of data to place in validation set
    @param validation_column: variable to perform validation on (if None then sample randomly)

    @return training and validation indices
    """

    if validation_column is None:
        # For no validation column just sample data randomly
        num_obs = len(obs_data)
        val_size = int(tuning_sample_ratio * num_obs)
        if tuning_sample_ratio > 0:
            val_size = max(val_size, 1)
        permuted_idx = np.random.permutation(num_obs)
        train_idx = permuted_idx[:num_obs - val_size]
        val_idx = permuted_idx[num_obs - val_size:]
    else:
        # For a validation column, sample the categories randomly based on
        # tuning_sample_ratio
        categories = set([elt[validation_column] for elt in metadata])
        num_categories = len(categories)
        val_size = int(tuning_sample_ratio * num_categories)
        if tuning_sample_ratio > 0:
            val_size = max(val_size, 1)

        # sample random categories from our validation variable
        val_categories = set(random.sample(categories, val_size))
        train_categories = categories - val_categories
        train_idx = [idx for idx, elt in enumerate(metadata) if elt[validation_column] in train_categories]
        val_idx = [idx for idx, elt in enumerate(metadata) if elt[validation_column] in val_categories]

    # construct training and validation sets based on CV regime above
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

    if tuning_sample_ratio > 0:
        assert(len(val_set) > 0)
    return train_set, val_set

def get_penalty_params(pen_param_str, solver):
    """
    @param pen_param_str: comma separated list of penalty parameters
    @param solver: the solver requested (L, FL, SFL)
    Determines the grid of penalty parameters to search over
    """
    penalty_params = [float(l) for l in pen_param_str.split(",")]
    sorted_pen_params = sorted(penalty_params, reverse=True)

    if solver == "SFL":
        # first param is lasso, second one is fused
        pen_params_lists = [[(p, p/i) for i in FUSED_LASSO_PENALTY_RATIO] for p in sorted_pen_params]
    elif solver == "FL":
        # first param is lasso, second one is fused
        pen_params_lists = [[(0, p) for p in sorted_pen_params]]
    else:
        pen_params_lists = [[(p,) for p in sorted_pen_params]]
    return pen_params_lists

def initialize_theta(theta_shape, theta_mask):
    """
    Initialize theta -- start with all zeros
    """
    theta = np.zeros(theta_shape)
    # Set the impossible thetas to -inf
    theta[~theta_mask] = -np.inf
    return theta

def do_validation_set_checks(theta, theta_mask, val_set, val_set_evaluator, feat_generator, true_theta, args):
    """
    Does various checks on the model fitted on the training data.
    Most importantly, it calculates the difference between the EM surrogate functions
    It also will calculate the marginal likelihood if args.chibs is True
    It will also compare against the true_theta if it is known and if the true_theta is the same shape
    """
    theta_err = None
    if true_theta is not None and true_theta.shape == theta.shape:
        theta_err = np.linalg.norm(true_theta[theta_mask] - theta[theta_mask])
        log.info("Difference between true and fitted theta %f" % theta_err)

    ll_chibs = None
    if args.chibs:
        val_chibs = LogLikelihoodEvaluator(
            val_set,
            feat_generator,
            num_jobs=args.num_jobs,
            scratch_dir=args.scratch_dir,
        )
        ll_chibs = val_chibs.get_log_lik(theta, burn_in=args.num_val_burnin)
        log.info("Chibs log likelihood estimate: %f" % ll_chibs)

    ll_ratio_lower_bound = None
    if val_set_evaluator is not None:
        log_lik_ratio, ll_ratio_lower_bound, upper_bound = val_set_evaluator.get_log_likelihood_ratio(theta)

    return ll_ratio_lower_bound, ll_chibs, theta_err

def main(args=sys.argv[1:]):
    args = parse_args()
    log.basicConfig(format="%(message)s", filename=args.log_file, level=log.DEBUG)
    np.random.seed(args.seed)

    feat_generator = SubmotifFeatureGenerator(motif_len=args.motif_len)

    log.info("Reading data")
    obs_data, metadata = read_gene_seq_csv_data(
            args.input_genes,
            args.input_seqs,
            motif_len=args.motif_len,
            sample=args.sample_regime,
            locus=args.locus,
            species=args.species,
        )
    train_set, val_set = create_train_val_sets(
            obs_data,
            feat_generator,
            metadata,
            args.tuning_sample_ratio,
            args.validation_column,
        )

    obs_seq_feat_base = []
    for obs_seq_mutation in obs_data:
        obs_seq_feat_base.append(feat_generator.create_base_features(obs_seq_mutation))
    log.info("Data statistics:")
    log.info("  Number of sequences: Train %d, Val %d" % (len(train_set), len(val_set)))
    log.info(get_data_statistics_print_lines(obs_data, feat_generator))
    log.info("Settings %s" % args)

    log.info("Running EM")

    motif_list = feat_generator.motif_list

    # Run EM on the lasso parameters from largest to smallest
    pen_params_lists = get_penalty_params(args.penalty_params, args.solver)

    theta_shape = (feat_generator.feature_vec_len, args.theta_num_col)
    theta_mask = get_possible_motifs_to_targets(motif_list, theta_shape)

    true_theta = None
    if args.theta_file != "":
        true_theta, _ = load_true_model(args.theta_file)

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
        scratch_dir=args.scratch_dir,
    )

    burn_in = args.burn_in
    results_list = []
    best_models = []
    for pen_params_list in pen_params_lists:
        best_model_in_list = None
        val_set_evaluator = None
        for penalty_params in pen_params_list:
            penalty_param_str = ",".join(map(str, penalty_params))
            log.info("Penalty parameter %s" % penalty_param_str)

            theta = initialize_theta(theta_shape, theta_mask)

            theta, _ = em_algo.run(
                theta=theta,
                burn_in=burn_in,
                penalty_params=penalty_params,
                max_em_iters=args.em_max_iters,
                fuse_center=args.fuse_center,
                train_and_val=False
            )
            burn_in = 0 # Only use burn in at the very beginning

            if args.tuning_sample_ratio > 0:
                # Do checks on the validation set
                log_lik_ratio, _, _ = do_validation_set_checks(
                    theta,
                    theta_mask,
                    val_set,
                    val_set_evaluator,
                    feat_generator,
                    true_theta,
                    args,
                )
                if log_lik_ratio is not None:
                    log.info("Comparing validation log likelihood for penalty param %s, ratio: %f" % (penalty_param_str, log_lik_ratio))

                if args.full_train:
                    theta, _ = em_algo.run(
                        theta=theta,
                        burn_in=burn_in,
                        penalty_params=penalty_params,
                        max_em_iters=args.em_max_iters,
                        train_and_val=True
                    )

            # Get the probabilities of the target nucleotides
            fitted_prob_vector = MultinomialSolver.solve(obs_data, feat_generator, theta) if not args.per_target_model else None
            curr_model_results = MethodResults(penalty_params, theta, fitted_prob_vector)

            # We save the final theta (potentially trained over all the data)
            results_list.append(curr_model_results)
            with open(args.intermediate_out_file, "w") as f:
                pickle.dump(results_list, f)

            log.info("==== FINAL theta, %s====" % curr_model_results)
            log.info(get_nonzero_theta_print_lines(theta, motif_list, feat_generator.motif_len))

            if best_model_in_list is None or log_lik_ratio > 0:
                best_model_in_list = curr_model_results
                log.info("===== Best model so far %s" % best_model_in_list)

                if val_set_evaluator is not None:
                    num_val_samples = val_set_evaluator.num_samples
                    val_set_evaluator.close()
                else:
                    num_val_samples = args.num_val_samples

                val_set_evaluator = LikelihoodComparer(
                    val_set,
                    feat_generator,
                    theta_ref=best_model_in_list.theta,
                    num_samples=num_val_samples,
                    burn_in=args.num_val_burnin,
                    num_jobs=args.num_jobs,
                    scratch_dir=args.scratch_dir,
                    num_threads=args.num_val_threads,
                )
            elif args.tuning_sample_ratio and log_lik_ratio < 0 and curr_model_results.num_nonzero > 0:
                # This model is not better than the previous model. Use a greedy approach and stop trying penalty parameters
                log.info("Stop trying penalty parameters for this penalty parameter list")
                break
        best_models.append(best_model_in_list)

    # A greedy comparison
    best_model = GreedyLikelihoodComparer.do_greedy_search(
        val_set,
        feat_generator,
        best_models,
        lambda m:get_num_unique_theta(m.theta),
        args.num_val_burnin,
        args.num_val_samples,
        args.num_jobs,
        args.scratch_dir,
    )

    log.info("=== FINAL Best model: %s" % best_model)
    if not args.full_train:
        log.info("Begin a final training of the model")
        # If we didn't do a full training for this best model, do it now
        best_theta, _ = em_algo.run(
            theta=best_model.theta,
            burn_in=burn_in,
            penalty_params=best_model.penalty_params,
            max_em_iters=args.em_max_iters,
            train_and_val=True
        )
        best_fitted_prob_vector = MultinomialSolver.solve(obs_data, feat_generator, best_theta) if not args.per_target_model else None
        best_model = MethodResults(
            best_model.penalty_params,
            best_theta,
            best_fitted_prob_vector,
        )

    with open(args.out_file, "w") as f:
        pickle.dump((best_model.theta, best_model.fitted_prob_vector), f)

if __name__ == "__main__":
    main(sys.argv[1:])
