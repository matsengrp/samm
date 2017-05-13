import numpy as np
from common import *

class MethodResults:
    def __init__(self, penalty_params):
        """
        @param penalized_theta: a theta from the penalized version
        """
        self.penalty_params = penalty_params

    def set_penalized_theta(self, penalized_theta, log_lik_ratio_lower_bound, log_lik_ratio, reference_model=None):
        """
        Store the model from the penalized stage
        """
        self.penalized_theta = penalized_theta
        self.log_lik_ratio = log_lik_ratio
        self.log_lik_ratio_lower_bound = log_lik_ratio_lower_bound
        if reference_model is not None:
            self.reference_penalty_params = reference_model.penalty_params
        else:
            self.reference_penalty_params = None
        self.penalized_num_nonzero = get_num_nonzero(self.penalized_theta)

    def set_refit_theta(self, refit_theta, variance_est, motifs_to_remove, motifs_to_remove_mask, possible_theta_mask, zero_theta_mask):
        """
        Store the model from the refit stage
        """
        self.refit_theta = refit_theta
        self.variance_est = variance_est
        self.motifs_to_remove = motifs_to_remove
        self.motifs_to_remove_mask = motifs_to_remove_mask
        self.refit_zero_theta_mask = zero_theta_mask
        self.refit_possible_theta_mask = possible_theta_mask
        self.zero_theta_mask = zero_theta_mask

    def __str__(self):
        pen_param_str = ",".join(map(str, self.penalty_params))
        return "Pen params %s" % pen_param_str
