import time
import numpy as np
import scipy as sp
import logging as log
from profile_support import profile

from common import *
from survival_problem_grad_descent import SurvivalProblemCustom

class SurvivalProblemProximal(SurvivalProblemCustom):
    min_diff_thres = 1e-10

    def solve_prox(self, theta, step_size):
        """
        Do proximal gradient step
        """
        raise NotImplementedError()

    def solve(self, init_theta, max_iters=1000, init_step_size=1, step_size_shrink=0.5, backtrack_alpha = 0.01, diff_thres=1e-6, min_iters=20, verbose=False):
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
        theta, current_value, upper_bound = self._solve(init_theta, max_iters, init_step_size, step_size_shrink, backtrack_alpha, diff_thres, min_iters, verbose)

        return theta, -current_value, -upper_bound

    def _get_value_parallel(self, theta):
        """
        Calculate the penalized loss value in parallel
        @return tuple: negative penalized log likelihood and the vector of log likelihoods
        """
        raise NotImplementedError()

    def _solve(self, init_theta, max_iters=100, init_step_size=1, step_size_shrink=0.5, backtrack_alpha = 0.01, diff_thres=1e-6, min_iters=10, verbose=False):
        """
        Runs proximal gradient descent to minimize the negative penalized log likelihood
        @return final fitted value of theta and penalized negative log likelihood and step size
        """
        st = time.time()
        theta = init_theta
        step_size = init_step_size
        diff = 0
        upper_bound = 0
        ase = None
        ess = None

        # Calculate loglikelihood of current theta
        init_value, log_lik_vec_init = self._get_value_parallel(theta)
        current_value = init_value

        for i in range(max_iters):
            if i % self.print_iter == 0:
                log.info("PROX iter %d, val %f, time %f" % (i, current_value, time.time() - st))
                if ase is not None and ess is not None:
                    log.info("  ase %f, ess %f" % (ase, ess))

            # Calculate gradient of the smooth part
            grad = self._get_gradient_log_lik(theta)
            potential_theta = theta - step_size * grad
            # Do proximal gradient step
            potential_theta = self.solve_prox(potential_theta, step_size)
            potential_value, potential_log_lik_vec = self._get_value_parallel(potential_theta)

            # Do backtracking line search
            expected_decrease = backtrack_alpha * np.power(np.linalg.norm(grad), 2)
            while potential_value >= current_value - step_size * expected_decrease:
                if step_size * expected_decrease < diff_thres and i > 0:
                    # Stop if difference in objective function is too small
                    break
                elif step_size * expected_decrease < self.min_diff_thres:
                    break
                step_size *= step_size_shrink
                log.info("PROX step size shrink %f" % step_size)
                potential_theta = theta - step_size * grad
                # Do proximal gradient step
                potential_theta = self.solve_prox(potential_theta, step_size)
                potential_value, potential_log_lik_vec = self._get_value_parallel(potential_theta)

            if potential_value > current_value:
                # Stop if value is increasing
                break
            else:
                # Calculate lower bound to determine if we need to rerun
                # Get the confidence interval around the penalized log likelihood (not the log likelihood itself!)
                log_lik_ratio_vec = potential_log_lik_vec - log_lik_vec_init
                ll_ratio_vec_grouped = self._group_log_lik_ratio_vec(log_lik_ratio_vec)
                # Upper bound becomes the lower bound when we consider the negative of this!
                ase, _, upper_bound, ess = get_standard_error_ci_corrected(ll_ratio_vec_grouped, ZSCORE, potential_value - init_value)

                # Calculate difference in objective function
                theta = potential_theta
                diff = current_value - potential_value
                current_value = potential_value

                if (upper_bound < 0 and i > min_iters) or diff < diff_thres:
                    # Stop if negative penalized log likelihood has significantly decreased and the minimum number of iters
                    # has been run or difference in objective function is small
                    break

        log.info("final PROX iter %d, val %f, time %d" % (i, current_value, time.time() - st))
        return theta, current_value, upper_bound
