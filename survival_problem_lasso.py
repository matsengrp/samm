import time
import numpy as np
import scipy as sp
import logging as log

from survival_problem_prox import SurvivalProblemProximal
from common import soft_threshold

class SurvivalProblemLasso(SurvivalProblemProximal):
    """
    Our own implementation of proximal gradient descent to solve the survival problem
    Objective function: - log likelihood of theta + lasso penalty on theta
    """
    min_diff_thres = 1e-8

    def solve_prox(self, theta, step_size):
        """
        Do proximal gradient step
        """
        new_theta_col0 = soft_threshold(theta[:,0:1], step_size * self.penalty_params[0])
        new_theta_target = soft_threshold(theta[:,1:], step_size * self.penalty_params[1])
        new_theta = np.hstack([new_theta_col0, new_theta_target])
        return new_theta

    def _get_value_parallel(self, theta):
        """
        @return tuple: negative penalized log likelihood and array of log likelihoods
        """
        log_lik_vec = self._get_log_lik_parallel(theta)
        neg_log_lik = -1.0/self.num_samples * log_lik_vec.sum()
        if self.possible_theta_mask is None:
            return neg_log_lik, log_lik_vec
        else:
            if self.penalty_params[1] == 0:
                l1_norm = np.linalg.norm(theta[self.possible_theta_mask], ord=1)
                return neg_log_lik + self.penalty_params[0] * l1_norm, log_lik_vec
            else:
                col0_mask = self.possible_theta_mask[:,0:1]
                theta_col0 = theta[:,0:1]
                l1_col0_norm = np.linalg.norm(theta_col0[col0_mask], ord=1)
                l1_norm_target = 0
                if self.per_target_model:
                    target_mask = self.possible_theta_mask[:,1:]
                    theta_target = theta[:,1:]
                    l1_norm_target = np.linalg.norm(theta_target[target_mask], ord=1)
                return neg_log_lik + self.penalty_params[0] * l1_col0_norm + self.penalty_params[1] * l1_norm_target, log_lik_vec
