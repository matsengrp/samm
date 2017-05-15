import pickle
import copy
import logging as log

from models import ObservedSequenceMutations
from mcmc_em import MCMC_EM
from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator
from method_results import MethodResults
from confidence_interval_maker import ConfidenceIntervalMaker
from model_truncation import ModelTruncation
from common import *

class ContextModelAlgo:
    """
    Performs the 2-stage fitting procedure
    """
    def __init__(self, feat_generator, obs_data, train_set, args, all_runs_pool, true_theta=None):
        """
        @param feat_generator: feature generator
        @param obs_data: full data set - used in training for the second stage
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
        )
        self.em_max_iters = args.em_max_iters

        self.true_theta = true_theta

        self.intermediate_out_dir = args.intermediate_out_dir
        self.motif_lens = args.motif_lens
        self.positions_mutating = args.positions_mutating
        self.z_stat = args.z_stat
        self.burn_in = args.burn_in

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

    def fit(self, penalty_param, val_set_evaluator=None, reference_pen_param=None):
        """
        @param penalty_param: penalty parameter for fitting the first stage
        @param val_set_evaluator: LikelihoodComparer with a given reference model
        @param reference_pen_param: the penalty parameters for the reference model

        @return the fitted model after the 2-step procedure
        """
        penalty_params = (penalty_param, )
        init_theta = initialize_theta(self.theta_shape, self.possible_theta_mask, self.zero_theta_mask)

        #### STAGE 1: FIT A PENALIZED MODEL
        penalized_theta, _, _ = self.em_algo.run(
            self.train_set,
            self.feat_generator,
            theta=init_theta,
            possible_theta_mask=self.possible_theta_mask,
            zero_theta_mask=self.zero_theta_mask,
            burn_in=self.burn_in,
            penalty_params=penalty_params,
            max_em_iters=self.em_max_iters,
            intermed_file_prefix="%s/e_samples_%f_" % (self.intermediate_out_dir, penalty_param),
        )
        curr_model_results = MethodResults(penalty_params, self.motif_lens, self.positions_mutating, self.z_stat)

        #### STAGE 1.5: DECIDE IF THIS MODEL IS WORTH REFITTING
        #### Right now, we check if the validation log likelihood (EM surrogate) is better
        log_lik_ratio_lower_bound, log_lik_ratio = self._do_validation_set_checks(
            penalized_theta,
            val_set_evaluator,
        )
        curr_model_results.set_penalized_theta(
            penalized_theta,
            log_lik_ratio_lower_bound,
            log_lik_ratio,
            reference_penalty_param=reference_pen_param,
        )

        log.info("==== Penalized theta, %f, nonzero %d ====" % (penalty_param, curr_model_results.penalized_num_nonzero))
        log.info(get_nonzero_theta_print_lines(penalized_theta, self.feat_generator))

        if log_lik_ratio_lower_bound is None or log_lik_ratio_lower_bound >= 0:
            # STAGE 2: REFIT THE MODEL WITH NO PENALTY
            model_masks = ModelTruncation(penalized_theta, feat_generator)
            log.info("Refit theta size: %d" % model_masks.zero_theta_mask_refit.size)
            if model_masks.zero_theta_mask_refit.size > 0:
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
                refit_theta, variance_est, _ = self.em_algo.run(
                    obs_data_stage2,
                    feat_generator_stage2,
                    theta=penalized_theta[~motifs_to_remove_mask,:], # initialize from the lasso version
                    possible_theta_mask=possible_theta_mask_refit,
                    zero_theta_mask=model_masks.zero_theta_mask_refit,
                    burn_in=self.burn_in,
                    penalty_params=(0,), # now fit with no penalty
                    max_em_iters=self.em_max_iters,
                    intermed_file_prefix="%s/e_samples_%f_full_" % (self.intermediate_out_dir, penalty_param),
                    get_hessian=True,
                )

                log.info("==== Refit theta, %s====" % curr_model_results)
                log.info(get_nonzero_theta_print_lines(refit_theta, feat_generator_stage2))

                num_not_crossing_zero = 0
                if variance_est is not None:
                    conf_int = ConfidenceIntervalMaker.create_confidence_intervals(
                        refit_theta,
                        np.sqrt(np.diag(variance_est)),
                        possible_theta_mask_refit,
                        zero_theta_mask_refit,
                        z=self.z_stat,
                    )
                    num_cross_zero = np.sum((conf_int[:,0] <= 0) & (0 <= conf_int[:,2]))
                    num_not_crossing_zero = conf_int.shape[0] - num_cross_zero

                curr_model_results.set_refit_theta(
                    refit_theta,
                    variance_est,
                    model_masks.feats_to_remove,
                    model_masks.feats_to_remove_mask,
                    possible_theta_mask_refit,
                    model_masks.zero_theta_mask_refit,
                    num_not_crossing_zero,
                )
                log.info("Pen_param %f, Number nonzero %d, Perc nonzero %f" % (penalty_param, curr_model_results.num_not_crossing_zero, curr_model_results.percent_not_crossing_zero))

        return curr_model_results
