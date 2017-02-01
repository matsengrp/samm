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
        self.fused_lasso_idx1 = np.array(motifs_fused_lasso1, dtype=int)
        self.fused_lasso_idx2 = np.array(motifs_fused_lasso2, dtype=int)

        # Creates the difference matrix. Won't work with too many motifs
        # TODO: Make more memory efficient in the future
        # self.D = np.matrix(np.zeros((self.fused_lasso_idx1.size, self.feature_generator.feature_vec_len)))
        # for i, (i1, i2) in enumerate(zip(self.fused_lasso_idx1.tolist(), self.fused_lasso_idx2.tolist())):
        #     self.D[i, i1] = 1
        #     self.D[i, i2] = -1

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
        beta = self.get_fused_lasso_theta(theta)
        u = np.matrix(np.zeros(beta.size)).T # scaled dual variable
        beta_size = beta.size
        eye_matrix = np.eye(beta_size)

        step_size = init_step_size
        current_value = self.get_value(theta)
        prev_value = current_value

        # Slowly increase the accuracy of the update to theta since it is an iterative algo
        gd_diff_thres = 0.3
        max_gd_iters = 2
        gd_step_size = 1
        for i in range(max_iters):
            if i % self.print_iter == 0:
                log.info("PROX Iter %d, val %f, time %d" % (i, current_value, time.time() - st))

            # Calculate gradient of the smooth part
            grad = self.get_gradient_smooth(theta)
            print "theta", theta
            print "grad", grad
            potential_theta = theta - step_size * grad

            # Do proximal gradient step
            # prox step 1: solve fused lasso
            potential_theta = self.solve_fused_lasso_prox(
                potential_theta,
                step_size * self.penalty_param_fused
            )
            print "potential_theta HAPPY", potential_theta
            # prox step 2: soft threshold to get sparse fused lasso
            potential_theta = soft_threshold(potential_theta, step_size * self.penalty_param)

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
                # prox step 1: solve fused lasso
                potential_theta = self.solve_fused_lasso_prox(potential_theta, step_size * self.penalty_param_fused)
                # prox step 2: soft threshold to get sparse fused lasso
                potential_theta = soft_threshold(potential_theta, step_size * self.penalty_param)

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
        return theta, current_value, step_size

    def solve_fused_lasso_prox(self, potential_theta, factor):
        """
        Minimize 0.5 * || theta - potential_theta ||^2_2 + factor * || fused lasso penalty on theta ||_1
        """
        print "factor", factor
        feat_list = self.motif_list + ["EDGES"]
        with open("fused_lasso_solver/in.txt", "w") as f:
            f.write("%f\n" % factor)
            for j, (motif, theta) in enumerate(zip(feat_list, potential_theta.tolist())):
                f.write("%s %d %f\n" % (motif, j, theta))

        st = time.time()
        os.system("./fused_lasso_solver/a.out")
        print " prox time",  time.time() - st

        with open("fused_lasso_solver/out.txt", "r") as f:
            lines = f.readlines()
        return np.array([float(l.split(" ")[1]) for l in lines])

    def get_gradient_smooth(self, theta):
        return self._get_gradient_log_lik(theta)

    def _get_value_parallel(self, theta):
        """
        @return negative penalized log likelihood
        """
        fused_lasso_pen = self.penalty_param_fused * np.linalg.norm(self.get_fused_lasso_theta(theta), ord=1)
        lasso_pen = self.penalty_param_lasso * np.linalg.norm(theta, ord=1)
        return -self._get_log_lik_parallel(theta) + fused_lasso_pen + lasso_pen
