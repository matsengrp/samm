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

    def get_value(self, theta):
        """
        @return negative penalized log likelihood
        """
        return -(1.0/self.num_samples * np.sum(self._get_log_lik_parallel(theta)) - self.penalty_param * np.linalg.norm(theta[self.theta_mask,], ord=1))

    def solve(self, init_theta, max_iters=1000, init_step_size=1, step_size_shrink=0.5, backtrack_alpha = 0.01, diff_thres=1e-6, verbose=False):
        """
        Runs proximal gradient descent to minimize the negative penalized log likelihood

        @param init_theta: where to initialize the gradient descent
        @param max_iters: maximum number of iterations of gradient descent
        @param init_step_size: how big to initialize the step size factor
        @param step_size_shrink: how much to shrink the step size during backtracking line descent
        @param backtrack_alpha: the alpha in backtracking line descent (p464 in Boyd)
        @param diff_thres: if the difference is less than diff_thres, then stop gradient descent
        @param verbose: whether to print out the status at each iteration
        @return final fitted value of theta and penalized log likelihood
        """
        theta, current_value, step_size = self._solve(init_theta, max_iters, init_step_size, step_size_shrink, backtrack_alpha, diff_thres, verbose)
        return theta, -current_value

    def solve_prox(self, theta, step_size):
        """
        Do proximal gradient step
        """
        return soft_threshold(theta, step_size * self.penalty_param)

    def _get_value_parallel(self, theta):
        """
        @return negative penalized log likelihood
        """
        return - (1.0/self.num_samples * np.sum(self._get_log_lik_parallel(theta)) - self.penalty_param * np.linalg.norm(theta[self.theta_mask,], ord=1))
