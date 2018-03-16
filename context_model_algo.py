import copy
import logging as log
import numpy as np
import scipy.stats

from mcmc_em import MCMC_EM
from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator
from method_results import MethodResults
from model_truncation import ModelTruncation
from common import *
from sampler_collection import SamplerCollection
from mutation_order_gibbs import MutationOrderGibbsSampler

class ContextModelAlgo:
    """
    Performs fitting procedures
    """
    def __init__(self, feat_generator, obs_data, train_set, args, all_runs_pool, true_theta=None):
        """
        @param feat_generator: feature generator
        @param obs_data: full data set - used in training for the refitting stage
        @param train_set: data just used for training the penalized model
        @param args: object with other useful settings (TODO: clean this up one day)
        @param all_runs_pool: multiprocessing pool
        @param true_theta: true theta if available
        """
        self.args = args
        self.feat_generator = feat_generator

        self.obs_data = obs_data
        self.train_set = train_set

        self.theta_shape = (feat_generator.feature_vec_len, args.theta_num_col)
        self.possible_theta_mask = get_possible_motifs_to_targets(
            feat_generator.motif_list,
            self.theta_shape,
            feat_generator.mutating_pos_list
        )
        self.zero_theta_mask = np.zeros(self.theta_shape, dtype=bool)

        self.em_algo = MCMC_EM(
            args.sampler_cls,
            args.problem_solver_cls,
            base_num_e_samples=args.num_e_samples,
            num_jobs=args.num_jobs,
            scratch_dir=args.scratch_dir,
            pool=all_runs_pool,
            per_target_model=args.per_target_model,
            sampling_rate=args.sampling_rate,
        )
        self.em_max_iters = args.em_max_iters

        self.true_theta = true_theta
        self.num_e_samples = args.num_e_samples
        self.num_jobs = args.num_jobs
        self.scratch_dir = args.scratch_dir
        self.intermediate_out_dir = args.intermediate_out_dir
        self.motif_lens = args.motif_lens
        self.positions_mutating = args.positions_mutating
        self.burn_in = args.burn_in
        self.sampling_rate = args.sampling_rate

    def _get_theta_err(self, theta, theta_mask):
        """
        Compares against the true_theta if it is known and if the true_theta is the same shape

        @param theta_mask: a mask with all the possible theta values (the ones that are not -inf)
        """
        theta_err = None
        if self.true_theta is not None and self.true_theta.shape == theta.shape:
            theta_err = np.linalg.norm(self.true_theta[theta_mask] - theta[theta_mask])
            if np.var(theta[theta_mask]) > 0:
                pearson_r, _ = scipy.stats.pearsonr(self.true_theta[theta_mask], theta[theta_mask])
                spearman_r, _ = scipy.stats.spearmanr(self.true_theta[theta_mask], theta[theta_mask])
                log.info("Difference between true and fitted theta %f, pear %f, spear %f" % (theta_err, pearson_r, spearman_r))
        return theta_err

    def _do_validation_set_checks(self, theta, val_set_evaluator):
        """
        @return the difference between the EM surrogate functions
        """
        ll_ratio_lower_bound = None
        log_lik_ratio = None
        if val_set_evaluator is not None:
            log_lik_ratio, ll_ratio_lower_bound, upper_bound = val_set_evaluator.get_log_likelihood_ratio(theta)
            log.info("Comparing validation log likelihood, log ratio: %f (lower bound: %f)" % (log_lik_ratio, ll_ratio_lower_bound))

        return ll_ratio_lower_bound, log_lik_ratio

    def fit_penalized(self, penalty_params, max_em_iters, val_set_evaluator=None, init_theta=None, reference_pen_param=None):
        """
        @param penalty_params: penalty parameter for fitting penalized model
        @param val_set_evaluator: LikelihoodComparer with a given reference model
        @param reference_pen_param: the penalty parameters for the reference model

        @return the fitted model after the 2-step procedure
        """
        if init_theta is None:
            init_theta = initialize_theta(self.theta_shape, self.possible_theta_mask, self.zero_theta_mask)

        penalized_theta, _, _, _ = self.em_algo.run(
            self.train_set,
            self.feat_generator,
            theta=init_theta,
            possible_theta_mask=self.possible_theta_mask,
            zero_theta_mask=self.zero_theta_mask,
            burn_in=self.burn_in,
            penalty_params=penalty_params,
            max_em_iters=max_em_iters,
            max_e_samples=self.num_e_samples * 4,
            intermed_file_prefix="%s/e_samples_%s_" % (self.intermediate_out_dir, "-".join([str(p) for p in penalty_params])),
        )
        curr_model_results = MethodResults(penalty_params, self.motif_lens, self.positions_mutating)

        #### Calculate validation log likelihood (EM surrogate), use to determine if model is any good.
        log_lik_ratio_lower_bound, log_lik_ratio = self._do_validation_set_checks(
            penalized_theta,
            val_set_evaluator,
        )
        curr_model_results.set_penalized_theta(
            penalized_theta,
            log_lik_ratio_lower_bound,
            log_lik_ratio,
            model_masks=ModelTruncation(penalized_theta, self.feat_generator),
            reference_penalty_param=reference_pen_param,
        )

        log.info("==== Penalized theta, %s, nonzero %d ====" % (penalty_params, curr_model_results.penalized_num_nonzero))
        log.info(get_nonzero_theta_print_lines(penalized_theta, self.feat_generator))
        return curr_model_results

    def refit_unpenalized(self, model_result, max_em_iters, get_hessian=True):
        """
        Refit the model
        Modifies model_result
        """
        model_masks = model_result.model_masks

        log.info("Refit theta size: %d" % model_masks.zero_theta_mask_refit.size)
        if model_masks.zero_theta_mask_refit.size == 0:
            return

        # Create a feature generator for this shrunken model
        feat_generator_stage2 = HierarchicalMotifFeatureGenerator(
            motif_lens=self.motif_lens,
            feats_to_remove=model_masks.feats_to_remove,
            left_motif_flank_len_list=self.positions_mutating,
        )
        # Get the data ready - using ALL data
        obs_data_stage2 = [copy.deepcopy(o) for o in self.obs_data]
        feat_generator_stage2.add_base_features_for_list(obs_data_stage2)
        # Create the theta mask for the shrunken theta
        possible_theta_mask_refit = get_possible_motifs_to_targets(
            feat_generator_stage2.motif_list,
            model_masks.zero_theta_mask_refit.shape,
            feat_generator_stage2.mutating_pos_list,
        )
        # Refit over the support from the penalized problem
        init_theta = model_result.penalized_theta[~model_masks.feats_to_remove_mask,:]
        init_theta[model_masks.zero_theta_mask_refit] = 0
        refit_theta, variance_est, sample_obs_info, _ = self.em_algo.run(
            obs_data_stage2,
            feat_generator_stage2,
            theta=init_theta, # initialize from the lasso version
            possible_theta_mask=possible_theta_mask_refit,
            zero_theta_mask=model_masks.zero_theta_mask_refit,
            burn_in=self.burn_in,
            penalty_params=(0,0), # now fit with no penalty
            max_em_iters=max_em_iters,
            max_e_samples=self.num_e_samples * 4,
            intermed_file_prefix="%s/e_samples_%f_full_" % (self.intermediate_out_dir, 0),
            get_hessian=get_hessian,
        )

        log.info("==== Refit theta, %s====" % model_result)
        log.info(get_nonzero_theta_print_lines(refit_theta, feat_generator_stage2))

        model_result.set_refit_theta(
            refit_theta,
            variance_est,
            sample_obs_info,
            possible_theta_mask_refit,
        )

    def _calculate_residuals_from_refit(self, model_result):
        model_masks = model_result.model_masks

        # Create a feature generator for this shrunken model
        feat_generator_stage2 = HierarchicalMotifFeatureGenerator(
            motif_lens=self.motif_lens,
            feats_to_remove=model_masks.feats_to_remove,
            left_motif_flank_len_list=self.positions_mutating,
        )
        # Get the data ready - using ALL data
        obs_data_stage2 = [copy.deepcopy(o) for o in self.obs_data]
        feat_generator_stage2.add_base_features_for_list(obs_data_stage2)

        sampler_collection = SamplerCollection(
            obs_data_stage2,
            model_result.refit_theta,
            MutationOrderGibbsSampler,
            feat_generator_stage2,
            self.num_jobs,
            self.scratch_dir,
            get_residuals=True,
        )
        init_orders = [
            np.random.permutation(obs_seq.mutation_pos_dict.keys()).tolist()
            for obs_seq in obs_data_stage2
        ]
        sampler_results = sampler_collection.get_samples(
            init_orders,
            self.num_e_samples,
            self.burn_in,
            sampling_rate=self.sampling_rate,
        )
        model_result.set_sampler_results(sampler_results)

    def _calculate_residuals_from_penalized(self, model_result):
        # Create a feature generator for this shrunken model
        feat_generator_stage2 = HierarchicalMotifFeatureGenerator(
            motif_lens=self.motif_lens,
            left_motif_flank_len_list=self.positions_mutating,
        )
        # Get the data ready - using ALL data
        obs_data_stage2 = [copy.deepcopy(o) for o in self.obs_data]
        feat_generator_stage2.add_base_features_for_list(obs_data_stage2)

        sampler_collection = SamplerCollection(
            obs_data_stage2,
            model_result.penalized_theta,
            MutationOrderGibbsSampler,
            feat_generator_stage2,
            self.num_jobs,
            self.scratch_dir,
            get_residuals=True,
        )
        init_orders = [
            np.random.permutation(obs_seq.mutation_pos_dict.keys()).tolist()
            for obs_seq in obs_data_stage2
        ]
        sampler_results = sampler_collection.get_samples(
            init_orders,
            self.num_e_samples,
            self.burn_in,
            sampling_rate=self.sampling_rate,
        )
        model_result.set_sampler_results(sampler_results)

    def calculate_residuals(self, model_result):
        """
        Similar to refit_unpenalized, but calculates residuals from refit theta
        Modifies model_result
        """

        if not model_result.has_refit_data:
            # no refit data to get residuals from
            self._calculate_residuals_from_penalized(model_result)
        else:
            self._calculate_residuals_from_refit(model_result)
