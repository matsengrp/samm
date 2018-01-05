import numpy as np
from common import get_num_nonzero

class MethodResults:
    def __init__(self, penalty_params, motif_lens, positions_mutating):
        """
        @param penalized_theta: a theta from the penalized version
        @param motif_lens: the motif lengths we fit for
        @param positions_mutating: the mutation positions that we fit for
        """
        self.penalty_params = penalty_params
        self.motif_lens = motif_lens
        self.positions_mutating = positions_mutating
        self.has_refit_data = False
        self.has_residuals = False
        self.num_not_crossing_zero = 0
        self.percent_not_crossing_zero = 1
        self.num_p = 0

    def set_penalized_theta(self, penalized_theta, log_lik_ratio_lower_bound, log_lik_ratio, model_masks, reference_penalty_param=None):
        """
        @param penalized_theta: the theta value after the first penalized stage
        @param log_lik_ratio_lower_bound: the lower bound of the confidence interval for the log lik ratio
        @param log_lik_ratio: the log lik ratio - as calculated using a LikelihoodComparer of a reference model
        @param reference_penalty_param: the penalty parameter for the reference model

        Store the model from the penalized stage
        """
        self.penalized_theta = penalized_theta
        self.log_lik_ratio = log_lik_ratio
        self.log_lik_ratio_lower_bound = log_lik_ratio_lower_bound
        self.reference_penalty_param = reference_penalty_param
        self.penalized_num_nonzero = get_num_nonzero(self.penalized_theta)
        self.model_masks = model_masks

    def set_refit_theta(self, refit_theta, variance_est, sample_obs_info, possible_theta_mask):
        """
        @param refit_theta: the theta value after the second refitting stage
        @param variance_est: a variance estimate if we were able to obtain one
        @param model_masks: ModelTruncation object
        @param possible_theta_mask: the theta values allowed to be a finite value for this shrunken refit theta (np bool array)

        Store the model from the refit stage
        """
        self.has_refit_data = True
        self.refit_theta = refit_theta
        self.variance_est = variance_est
        self.sample_obs_info = sample_obs_info
        self.refit_possible_theta_mask = possible_theta_mask
        self.num_p = np.sum(self.refit_possible_theta_mask & ~self.model_masks.zero_theta_mask_refit)

    def set_sampler_results(self, sampler_results):
        """
        Store the residuals
        """
        self.has_residuals = True
        self.sampler_results = sampler_results

    def __str__(self):
        pen_param_str = ",".join(map(str, self.penalty_params))
        return "Pen params %s" % pen_param_str
