import numpy as np
from common import *

class MethodResults:
    def __init__(self, penalty_params, motif_lens, positions_mutating, z_stat):
        """
        @param penalized_theta: a theta from the penalized version
        @param motif_lens: the motif lengths we fit for
        @param positions_mutating: the mutation positions that we fit for
        @param z_stat: the confidence interval width used (SE * z_stat)
        """
        self.penalty_params = penalty_params
        self.motif_lens = motif_lens
        self.positions_mutating = positions_mutating
        self.has_refit_data = False
        self.z_stat = z_stat
        self.num_not_crossing_zero = 0

    def set_penalized_theta(self, penalized_theta, log_lik_ratio_lower_bound, log_lik_ratio, reference_penalty_param=None):
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

    def set_refit_theta(self, refit_theta, variance_est, motifs_to_remove, motifs_to_remove_mask, possible_theta_mask, zero_theta_mask, num_not_crossing_zero):
        """
        @param refit_theta: the theta value after the second refitting stage
        @param variance_est: a variance estimate if we were able to obtain one
        @param motifs_to_remove: the motifs we removed when we fit this model (list of strings)
        @param motifs_to_remove_mask: the motifs we removed, but a numpy boolean mask of the full theta
        @param possible_theta_mask: the theta values allowed to be a finite value for this shrunken refit theta (np bool array)
        @param zero_theta_mask: the theta values we force to be zero in this shrinken refit theta (np bool array)
        @param num_not_crossing_zero: number of confidence intervals that didnt include zero, using z_stat for the width

        Store the model from the refit stage
        """
        self.has_refit_data = True
        self.refit_theta = refit_theta
        self.variance_est = variance_est
        self.motifs_to_remove = motifs_to_remove
        self.motifs_to_remove_mask = motifs_to_remove_mask
        self.refit_zero_theta_mask = zero_theta_mask
        self.refit_possible_theta_mask = possible_theta_mask
        self.zero_theta_mask = zero_theta_mask
        self.num_not_crossing_zero = num_not_crossing_zero

    def __str__(self):
        pen_param_str = ",".join(map(str, self.penalty_params))
        return "Pen params %s" % pen_param_str
