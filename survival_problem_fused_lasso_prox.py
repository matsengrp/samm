from cvxpy import *
import time
import numpy as np
import logging as log
from multiprocessing import Pool

from survival_problem_prox import SurvivalProblemProximal
from common import *
from solver_wrappers import solve_fused_lasso

class SurvivalProblemFusedLassoProximal(SurvivalProblemProximal):
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
        return -(self._get_log_lik_parallel(theta) - self.penalty_param_fused * fused_lasso_pen - self.penalty_param_lasso * lasso_pen)

    def get_fused_lasso_theta(self, theta):
        """
        @return the components of the fused lasso penalty (before applying l1 to it)
        """
        return theta[self.fused_lasso_idx1] - theta[self.fused_lasso_idx2]

    def solve_prox(self, theta, step_size):
        """
        Do proximal gradient step
        """
        # prox step 1: solve fused lasso
        fused_lasso_theta = self.solve_fused_lasso_prox(theta.reshape((theta.size,)), step_size * self.penalty_param_fused)
        # prox step 2: soft threshold to get sparse fused lasso
        sparse_fused_lasso_theta = soft_threshold(fused_lasso_theta, step_size * self.penalty_param_lasso)
        return sparse_fused_lasso_theta.reshape((sparse_fused_lasso_theta.size,1))

    def solve_fused_lasso_prox_cvxpy(self, potential_theta, factor):
        """
        Only used for checking against solve_fused_lasso_prox

        solve the fused lasso proximal problem using cvxpy
        Minimize 0.5 * || theta - potential_theta ||^2_2 + factor * || fused lasso penalty on theta ||_1
        """
        theta_var = Variable(potential_theta.size)
        obj = sum_squares(theta_var - potential_theta)

        fused_lasso_pen = 0
        for i1, i2 in zip(self.fused_lasso_idx1, self.fused_lasso_idx2):
            fused_lasso_pen += abs(theta_var[i1] - theta_var[i2])

        st = time.time()
        problem = Problem(Minimize(0.5 * obj + factor * fused_lasso_pen))
        problem.solve(verbose=True)
        return np.array(theta_var.value.flat)

    def solve_fused_lasso_prox(self, potential_theta, factor):
        """
        Solve the fused lasso proximal problem using the C library
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

    def _get_value_parallel(self, theta):
        """
        @return negative penalized log likelihood
        """
        fused_lasso_pen = self.penalty_param_fused * np.linalg.norm(self.get_fused_lasso_theta(theta), ord=1)
        lasso_pen = self.penalty_param_lasso * np.linalg.norm(theta, ord=1)
        return -self._get_log_lik_parallel(theta) + fused_lasso_pen + lasso_pen
