import time
import numpy as np
import scipy as sp
import logging as log
from multiprocessing import Pool

from survival_problem_prox import SurvivalProblemProximal
from common import soft_threshold

class SurvivalProblemLasso(SurvivalProblemProximal):
    """
    Our own implementation of proximal gradient descent to solve the survival problem
    Objective function: - log likelihood of theta + lasso penalty on theta
    """
    min_diff_thres = 1e-8

    def post_init(self):
        self.penalty_param = self.penalty_params[0]

    def get_value(self, theta):
        """
        @return negative penalized log likelihood
        """
        l1_norm = np.linalg.norm(theta[self.theta_mask,], ord=1) if self.theta_mask is not None else 0
        return -1.0/self.num_samples * np.sum(self._get_log_lik_parallel(theta)) + self.penalty_param * l1_norm

    def solve_prox(self, theta, step_size):
        """
        Do proximal gradient step
        """
        return soft_threshold(theta, step_size * self.penalty_param)

    def _get_value_parallel(self, theta):
        """
        @return tuple: negative penalized log likelihood and array of log likelihoods
        """
        log_lik_vec = self._get_log_lik_parallel(theta)
        lasso_pen = np.linalg.norm(theta[self.theta_mask,], ord=1) if self.theta_mask is not None else 0
        return -1.0/self.num_samples * log_lik_vec.sum() + self.penalty_param * lasso_pen, log_lik_vec
