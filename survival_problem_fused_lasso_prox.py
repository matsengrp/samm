import time
import os
import numpy as np
import scipy as sp
import sklearn.linear_model
import logging as log
from multiprocessing import Pool

from survival_problem_grad_descent import SurvivalProblemCustom
from survival_problem_lasso import SurvivalProblemLasso
from common import *
from solver_wrappers import solve_fused_lasso

class SurvivalProblemFusedLassoProximal(SurvivalProblemCustom):
    """
    Let's do proximal gradient descent to solve the sparse fused lasso problem.
    In this case we "fuse" over motifs that differ in only one base.
    """
    print_iter = 1
    min_diff_thres = 1e-6

    def post_init(self):
        # Calculate the fused lasso indices
        self.motif_list = self.feature_generator.get_motif_list()

        # We implement the fused penalty in terms of differences of pairs that are stored in these
        # index lists: the first entry of the first list minus the first entry in the second list, etc.
        motifs_fused_lasso1 = []
        motifs_fused_lasso2 = []
        for i1, m1 in enumerate(self.motif_list):
            for i2, m2 in enumerate(self.motif_list):
                if i1 == i2:
                    continue
                if get_idx_differ_by_one_character(m1, m2) is not None:
                    motifs_fused_lasso1.append(i1)
                    motifs_fused_lasso2.append(i2)
        self.fused_lasso_idx1 = np.array(motifs_fused_lasso1, dtype=np.intc)
        self.fused_lasso_idx2 = np.array(motifs_fused_lasso2, dtype=np.intc)

        self.penalty_param_fused = self.penalty_param
        # TODO: This is a hack for now since we assume only one penalty param
        # We upweight lasso since we don't want to over-penalize.
        self.penalty_param_lasso = 2 * self.penalty_param

    def get_value(self, theta):
        """
        @return negative penalized log likelihood
        """
        fused_lasso_pen = np.linalg.norm(self.get_fused_lasso_theta(theta), ord=1)
        lasso_pen = np.linalg.norm(theta, ord=1)
        return -(self.get_log_lik(theta) - self.penalty_param_fused * fused_lasso_pen - self.penalty_param_lasso * lasso_pen)

    def get_fused_lasso_theta(self, theta):
        """
        @return the components of the fused lasso penalty (before applying l1 to it)
        """
        return theta[self.fused_lasso_idx1] - theta[self.fused_lasso_idx2]

    def solve(self, init_theta, max_iters=1000, num_threads=1, init_step_size=1.0, step_size_shrink=0.5, backtrack_alpha = 0.01, diff_thres = 1e-6, verbose=False):
        self.pool = Pool(num_threads)
        st = time.time()
        theta = init_theta

        step_size = init_step_size
        current_value = self.get_value(theta)
        # Slowly increase the accuracy of the update to theta since it is an iterative algo
        for i in range(max_iters):
            if i % self.print_iter == 0:
                log.info("PROX Iter %d, val %f, time %d" % (i, current_value, time.time() - st))

            # Calculate gradient of the smooth part
            grad = self.get_gradient_smooth(theta)
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
                log.info("GD step size shrink %f" % step_size)

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
        log.info("final GD iter %d, val %f, time %d" % (i, current_value, time.time() - st))
        return theta, -current_value

    def solve_prox(self, theta, step_size):
        """
        Do proximal gradient step
        """

        # prox step 1: solve fused lasso
        fused_lasso_theta = self.solve_fused_lasso_prox(theta, step_size * self.penalty_param_fused)
        # prox step 2: soft threshold to get sparse fused lasso
        sparse_fused_lasso_theta = soft_threshold(fused_lasso_theta, step_size * self.penalty_param_lasso)
        return sparse_fused_lasso_theta

    def solve_fused_lasso_prox(self, potential_theta, factor):
        """
        Minimize 0.5 * || theta - potential_theta ||^2_2 + factor * || fused lasso penalty on theta ||_1
        """
        fused_lasso_soln = np.zeros(potential_theta.size)
        solve_fused_lasso(
            fused_lasso_soln,
            np.array(potential_theta, dtype=np.float64),
            self.fused_lasso_idx1,
            self.fused_lasso_idx2,
            factor,
        )
        return fused_lasso_soln

    def get_gradient_smooth(self, theta):
        return self._get_gradient_log_lik(theta)

    def _get_value_parallel(self, theta):
        """
        @return negative penalized log likelihood
        """
        fused_lasso_pen = self.penalty_param_fused * np.linalg.norm(self.get_fused_lasso_theta(theta), ord=1)
        lasso_pen = self.penalty_param_lasso * np.linalg.norm(theta, ord=1)
        return -self._get_log_lik_parallel(theta) + fused_lasso_pen + lasso_pen
