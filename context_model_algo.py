import pickle

from models import ObservedSequenceMutations
from mcmc_em import MCMC_EM
from submotif_feature_generator import SubmotifFeatureGenerator
from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator
from mutation_order_gibbs import MutationOrderGibbsSampler
from survival_problem_lasso import SurvivalProblemLasso
from likelihood_evaluator import *
from multinomial_solver import MultinomialSolver
from method_results import MethodResults
from confidence_interval_maker import ConfidenceIntervalMaker
from common import *

class ContextModelAlgo:
    def __init__(self, feat_generator, obs_data, train_set, val_set, args, all_runs_pool):
        self.args = args
        self.feat_generator = feat_generator
        self.obs_data = obs_data
        self.train_set = train_set
        self.val_set = val_set
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
        self.intermediate_out_dir = args.intermediate_out_dir
        self.motif_lens = args.motif_lens
        self.positions_mutating = args.positions_mutating
        self.out_file = args.out_file
        self.z_stat = args.z_stat
        self.burn_in = args.burn_in

    def fit(self, penalty_param):
        penalty_params = (penalty_param, )
        init_theta = initialize_theta(self.theta_shape, self.possible_theta_mask, self.zero_theta_mask)
        base_train_obs = self.feat_generator.create_base_features_for_list(self.train_set)
        base_val_obs = self.feat_generator.create_base_features_for_list(self.val_set)

        #### STAGE 1: FIT A PENALIZED MODEL
        penalized_theta, variance_est, _ = self.em_algo.run(
            base_train_obs,
            self.feat_generator,
            theta=init_theta,
            possible_theta_mask=self.possible_theta_mask,
            zero_theta_mask=self.zero_theta_mask,
            burn_in=self.burn_in,
            penalty_params=penalty_params,
            max_em_iters=self.em_max_iters,
            intermed_file_prefix="%s/e_samples_%f_" % (self.intermediate_out_dir, penalty_param),
        )
        curr_model_results = MethodResults(penalty_params, self.motif_lens, self.positions_mutating)
        curr_model_results.set_penalized_theta(penalized_theta, None, None, reference_model=None)

        log.info("==== Penalized theta, %f, nonzero %d ====" % (penalty_param, curr_model_results.penalized_num_nonzero))
        log.info(get_nonzero_theta_print_lines(penalized_theta, self.feat_generator))

        # STAGE 2: REFIT THE MODEL WITH NO PENALTY
        zero_theta_mask_refit, motifs_to_remove, motifs_to_remove_mask = make_zero_theta_refit_mask(
            penalized_theta,
            self.feat_generator,
        )
        curr_model_results.zero_theta_mask_refit = zero_theta_mask_refit
        log.info("Refit theta size: %d" % zero_theta_mask_refit.size)
        if zero_theta_mask_refit.size > 0:
            # Create a feature generator for this shrunken model
            feat_generator_stage2 = HierarchicalMotifFeatureGenerator(
                motif_lens=self.motif_lens,
                motifs_to_remove=motifs_to_remove,
                left_motif_flank_len_list=self.positions_mutating,
            )
            # Get the data ready - using ALL data
            obs_data_stage2 = feat_generator_stage2.create_base_features_for_list(self.obs_data)
            # Create the theta mask for the shrunken theta
            possible_theta_mask_refit = get_possible_motifs_to_targets(
                feat_generator_stage2.motif_list,
                zero_theta_mask_refit.shape,
                feat_generator_stage2.mutating_pos_list,
            )
            # Refit over the support from the penalized problem
            refit_theta, variance_est, _ = self.em_algo.run(
                obs_data_stage2,
                feat_generator_stage2,
                theta=penalized_theta[~motifs_to_remove_mask,:], # initialize from the lasso version
                possible_theta_mask=possible_theta_mask_refit,
                zero_theta_mask=zero_theta_mask_refit,
                burn_in=self.burn_in,
                penalty_params=(0,), # now fit with no penalty
                max_em_iters=self.em_max_iters,
                intermed_file_prefix="%s/e_samples_%f_full_" % (self.intermediate_out_dir, penalty_param),
                get_hessian=True,
            )
            curr_model_results.set_refit_theta(
                refit_theta,
                variance_est,
                motifs_to_remove,
                motifs_to_remove_mask,
                possible_theta_mask_refit,
                zero_theta_mask_refit,
            )

            log.info("==== Refit theta, %s====" % curr_model_results)
            log.info(get_nonzero_theta_print_lines(refit_theta, feat_generator_stage2))

            out_file = self.out_file.replace(".pkl", "%f.pkl" % penalty_param)
            print "out_file", out_file
            with open(out_file, "w") as f:
                pickle.dump(curr_model_results, f)
        return curr_model_results

    def fit_for_opt(self, penalty_param):
        print "penalty_param", penalty_param
        curr_model_results = self.fit(penalty_param)
        if curr_model_results.zero_theta_mask_refit.size > 0 and curr_model_results.variance_est is not None:
            conf_int = ConfidenceIntervalMaker.create_confidence_intervals(
                curr_model_results.refit_theta,
                np.sqrt(np.diag(curr_model_results.variance_est)),
                curr_model_results.refit_possible_theta_mask,
                curr_model_results.refit_zero_theta_mask,
                z=self.z_stat,
            )
            num_cross_zero = np.sum((conf_int[:,0] <= 0) & (0 <= conf_int[:,2]))
            print "num_cross_zero", num_cross_zero
            return conf_int.shape[0] - num_cross_zero
        else:
            print "num_cross_zero", 0
            return 0
