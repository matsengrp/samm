import time
import numpy as np
import scipy as sp
import logging as log
from multiprocessing import Pool

from survival_problem_grad_descent import SurvivalProblemCustom

class SurvivalProblemProximal(SurvivalProblemCustom):
    min_diff_thres = 1e-10

    def solve_prox(self, theta, step_size):
        """
        Do proximal gradient step
        """
        raise NotImplementedError()

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

    def _get_value_parallel(self, theta):
        """
        Calculate the penalized loss value in parallel
        """
        raise NotImplementedError()

    def _solve(self, init_theta, max_iters=1000, init_step_size=1, step_size_shrink=0.5, backtrack_alpha = 0.01, diff_thres=1e-6, verbose=False):
        """
        Runs proximal gradient descent to minimize the negative penalized log likelihood
        @return final fitted value of theta and penalized negative log likelihood and step size
        """
        self.pool = Pool(self.num_threads)

        st = time.time()
        theta = init_theta
        step_size = init_step_size
        current_value = self._get_value_parallel(theta)
        for i in range(max_iters):
            if i % self.print_iter == 0:
                log.info("PROX iter %d, val %f, time %f" % (i, current_value, time.time() - st))

            # Calculate gradient of the smooth part
            grad = self._get_gradient_log_lik(theta)
            potential_theta = theta - step_size * grad
            # Do proximal gradient step
            potential_theta = self.solve_prox(potential_theta, step_size)
            potential_value = self._get_value_parallel(potential_theta)

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
                potential_value = self._get_value_parallel(potential_theta)

            if potential_value > current_value:
                # Stop if value is increasing
                break
            else:
                theta = potential_theta
                diff = current_value - potential_value
                current_value = potential_value
                if diff < diff_thres:
                    # Stop if difference in objective function is too small
                    break

        self.pool.close()
        log.info("final PROX iter %d, val %f, time %d" % (i, current_value, time.time() - st))
        return theta, current_value, step_size