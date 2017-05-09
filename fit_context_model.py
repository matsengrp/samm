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
from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator
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
    parser.add_argument('--solver',
        type=str,
        help='CL = cvxpy lasso, CFL = cvxpy fused lasso, L = gradient descent lasso, FL = fused lasso, SFL = sparse fused lasso,',
        choices=["CL", "CFL", "L", "FL", "SFL"],
        default="L")
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
        default="1, 0.1, 0.01")
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
    elif args.solver == "FL" or args.solver == "SFL":
        if args.per_target_model:
            raise NotImplementedError()
        else:
            args.problem_solver_cls = SurvivalProblemFusedLassoProximal

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

def split_train_val_sets(obs_data, feat_generator, metadata, tuning_sample_ratio, validation_column):
    """
    @param feat_generator: submotif feature generator
    @param metadata: metadata to include variables to perform validation on
    @param tuning_sample_ratio: ratio of data to place in validation set
    @param validation_column: variable to perform validation on (if None then sample randomly)

    @return training and validation indices
    """
    num_obs = len(obs_data)
    if validation_column is None:
        # For no validation column just sample data randomly
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
        log.info("train_categories %s" % train_categories)
        log.info("val_categories %s" % val_categories)
        train_idx = [idx for idx, elt in enumerate(metadata) if elt[validation_column] in train_categories]
        val_idx = [idx for idx, elt in enumerate(metadata) if elt[validation_column] in val_categories]

    train_set = [obs_data[i] for i in train_idx]
    val_set = [obs_data[i] for i in val_idx]
    return train_set, val_set

def get_penalty_params(pen_param_str, solver):
    """
    @param pen_param_str: comma separated list of penalty parameters
    @param solver: the solver requested (L, FL, SFL)
    Determines the grid of penalty parameters to search over
    """
    penalty_params = [float(l) for l in pen_param_str.split(",")]
    sorted_pen_params = sorted(penalty_params, reverse=True)
    return [(p,) for p in sorted_pen_params]

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

def do_validation_set_checks(theta, theta_mask, val_set, val_set_evaluator, feat_generator, true_theta):
    """
    Does various checks on the model fitted on the training data.
    Most importantly, it calculates the difference between the EM surrogate functions
    It will also compare against the true_theta if it is known and if the true_theta is the same shape

    @param theta_mask: a mask with all the possible theta values (the ones that are not -inf)
    """
    theta_err = None
    if true_theta is not None and true_theta.shape == theta.shape:
        theta_err = np.linalg.norm(true_theta[theta_mask] - theta[theta_mask])
        pearson_r, _ = scipy.stats.pearsonr(true_theta[theta_mask], theta[theta_mask])
        spearman_r, _ = scipy.stats.spearmanr(true_theta[theta_mask], theta[theta_mask])
        log.info("Difference between true and fitted theta %f, pear %f, spear %f" % (theta_err, pearson_r, spearman_r))

    ll_ratio_lower_bound = None
    log_lik_ratio = None
    if val_set_evaluator is not None:
        log_lik_ratio, ll_ratio_lower_bound, upper_bound = val_set_evaluator.get_log_likelihood_ratio(theta)
        log.info("Comparing validation log likelihood, log ratio: %f (lower bound: %f)" % (log_lik_ratio, ll_ratio_lower_bound))

    return ll_ratio_lower_bound, log_lik_ratio, theta_err

def main(args=sys.argv[1:]):
    args = parse_args()
    log.basicConfig(format="%(message)s", filename=args.log_file, level=log.DEBUG)
    np.random.seed(args.seed)

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

    train_set, val_set = split_train_val_sets(
        obs_data,
        feat_generator,
        metadata,
        args.tuning_sample_ratio,
        args.validation_column,
    )
    base_train_obs = feat_generator.create_base_features_for_list(train_set)
    base_val_obs = feat_generator.create_base_features_for_list(val_set)

    log.info("Data statistics:")
    log.info("  Number of sequences: Train %d, Val %d" % (len(train_set), len(val_set)))
    log.info("Settings %s" % args)

    log.info("Running EM")

    # Run EM on the lasso parameters from largest to smallest
    pen_params_list = get_penalty_params(args.penalty_params, args.solver)

    theta_shape = (feat_generator.feature_vec_len, args.theta_num_col)
    possible_theta_mask = get_possible_motifs_to_targets(
        feat_generator.motif_list,
        theta_shape,
        feat_generator.mutating_pos_list
    )
    zero_theta_mask = np.zeros(theta_shape, dtype=bool)

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
        per_target_model=args.per_target_model,
    )

    burn_in = args.burn_in
    num_val_samples = args.num_val_samples
    results_list = []
    val_set_evaluator = None
    best_model = None
    for penalty_params in pen_params_list:
        penalty_param_str = ",".join(map(str, penalty_params))
        log.info("==== Penalty parameter %s ====" % penalty_param_str)

        init_theta = initialize_theta(theta_shape, possible_theta_mask, zero_theta_mask)

        #### STAGE 1: FIT A PENALIZED MODEL
        penalized_theta, variance_est, _ = em_algo.run(
            base_train_obs,
            feat_generator,
            theta=init_theta,
            possible_theta_mask=possible_theta_mask,
            zero_theta_mask=zero_theta_mask,
            burn_in=burn_in,
            penalty_params=penalty_params,
            max_em_iters=args.em_max_iters,
            intermed_file_prefix="%s/e_samples_%s_" % (args.intermediate_out_dir, penalty_param_str),
        )
        curr_model_results = MethodResults(penalty_params)

        #### STAGE 1.5: DECIDE IF THIS MODEL IS WORTH REFITTING
        #### Right now, we check if the validation log likelihood (EM surrogate) is better
        log_lik_ratio_lower_bound, log_lik_ratio, _ = do_validation_set_checks(
            penalized_theta,
            possible_theta_mask,
            base_val_obs,
            val_set_evaluator,
            feat_generator,
            true_theta,
        )
        curr_model_results.set_penalized_theta(penalized_theta, log_lik_ratio_lower_bound, log_lik_ratio, reference_model=best_model)

        log.info("==== Penalized theta, %s, nonzero %d ====" % (penalty_param_str, curr_model_results.penalized_num_nonzero))
        log.info(get_nonzero_theta_print_lines(penalized_theta, feat_generator))

        if log_lik_ratio_lower_bound is None or log_lik_ratio_lower_bound >= 0:
            # If the model is better than previous models
            best_model = curr_model_results
            log.info("===== Best model (per validation set) %s" % best_model)

            # Create this val set evaluator for next time
            val_set_evaluator = LikelihoodComparer(
                base_val_obs,
                feat_generator,
                theta_ref=best_model.penalized_theta,
                num_samples=num_val_samples,
                burn_in=args.num_val_burnin,
                num_jobs=args.num_jobs,
                scratch_dir=args.scratch_dir,
                pool=all_runs_pool,
            )
            # grab this many validation samples from now on
            num_val_samples = val_set_evaluator.num_samples

            # STAGE 2: REFIT THE MODEL WITH NO PENALTY
            zero_theta_mask_refit, motifs_to_remove = make_zero_theta_refit_mask(
                penalized_theta,
                feat_generator,
            )
            log.info("Refit theta size: %d" % zero_theta_mask_refit.size)
            if zero_theta_mask_refit.size > 0:
                # Create a feature generator for this shrunken model
                feat_generator_stage2 = HierarchicalMotifFeatureGenerator(
                    motif_lens=args.motif_lens,
                    motifs_to_remove=motifs_to_remove,
                    left_motif_flank_len_list=args.positions_mutating,
                )
                # Get the data ready - using ALL data
                obs_data_stage2 = feat_generator_stage2.create_base_features_for_list(obs_data)
                # Create the theta mask for the shrunken theta
                possible_theta_mask_refit = get_possible_motifs_to_targets(
                    feat_generator_stage2.motif_list,
                    zero_theta_mask_refit.shape,
                    feat_generator_stage2.mutating_pos_list,
                )
                # Refit over the support from the penalized problem
                refit_theta, variance_est, _ = em_algo.run(
                    obs_data_stage2,
                    feat_generator_stage2,
                    theta=penalized_theta, # initialize from the lasso version
                    possible_theta_mask=possible_theta_mask_refit,
                    zero_theta_mask=zero_theta_mask_refit,
                    burn_in=burn_in,
                    penalty_params=(0,), # now fit with no penalty
                    max_em_iters=args.em_max_iters,
                    intermed_file_prefix="%s/e_samples_%s_full_" % (args.intermediate_out_dir, penalty_param_str),
                    get_hessian=True,
                )
                curr_model_results.set_refit_theta(refit_theta, variance_est, motifs_to_remove, zero_theta_mask_refit)

                log.info("==== Refit theta, %s====" % curr_model_results)
                log.info(get_nonzero_theta_print_lines(refit_theta, feat_generator_stage2))

        # Save model results
        results_list.append(curr_model_results)
        with open(args.out_file, "w") as f:
            pickle.dump(results_list, f)

        if log_lik_ratio_lower_bound is not None and log_lik_ratio_lower_bound < 0 and curr_model_results.penalized_num_nonzero > 0:
            # This model is not better than the previous model. Use a greedy approach and stop trying penalty parameters
            log.info("Stop trying penalty parameters for this penalty parameter list")
            break

    if all_runs_pool is not None:
        all_runs_pool.close()
        # helpful comment copied over: make sure we don't keep these processes open!
        all_runs_pool.join()

if __name__ == "__main__":
    main(sys.argv[1:])
