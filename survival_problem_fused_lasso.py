import time
import numpy as np
import scipy as sp
from multiprocessing import Pool
import sklearn.linear_model
import logging as log

from survival_problem_grad_descent import SurvivalProblemCustom
from survival_problem_cvxpy import SurvivalProblemIshLassoCVXPY, SurvivalProblemFusedLassoCVXPY
from common import *

class SurvivalProblemFusedLasso(SurvivalProblemCustom):
    """
    Let's do ADMM?
    """
    print_iter = 1

    def post_init(self):
        self.penalty_param_fused = self.penalty_param
        self.penalty_param_lasso = self.penalty_param

        # Calculate the fused lasso indices
        motif_list = self.feature_generator.get_motif_list()
        fused_lasso_pen = 0
        motifs_fused_lasso1 = []
        motifs_fused_lasso2 = []
        for i1, m1 in enumerate(motif_list):
            for i2, m2 in enumerate(motif_list):
                if i1 == i2:
                    continue
                if get_idx_differ_by_one_character(m1, m2) is not None:
                    motifs_fused_lasso1.append(i1)
                    motifs_fused_lasso2.append(i2)
        self.fused_lasso_idx1 = np.array(motifs_fused_lasso1, dtype=int)
        self.fused_lasso_idx2 = np.array(motifs_fused_lasso2, dtype=int)

        self.D = np.matrix(np.zeros((self.fused_lasso_idx1.size, self.feature_generator.feature_vec_len)))
        for i, (i1, i2) in enumerate(zip(self.fused_lasso_idx1.tolist(), self.fused_lasso_idx2.tolist())):
            self.D[i, i1] = 1
            self.D[i, i2] = -1
        self.DD = self.D.T * self.D

    def get_value(self, theta):
        """
        @return negative penalized log likelihood
        """
        fused_lasso_pen = np.linalg.norm(self.get_fused_lasso_theta(theta), ord=1)
        lasso_pen = np.linalg.norm(theta, ord=1)
        return -(self.get_log_lik(theta) - self.penalty_param_fused * fused_lasso_pen - self.penalty_param_lasso * lasso_pen)

    def get_fused_lasso_theta(self, theta):
        return theta[self.fused_lasso_idx1, 0] - theta[self.fused_lasso_idx2, 0]

    def solve(self, init_theta, max_iters=1000, num_threads=1, init_step_size=1.0, step_size_shrink=0.5, diff_thres=1e-6, verbose=False):
        st = time.time()
        theta = np.matrix(init_theta).T
        print "theta", theta
        beta = self.get_fused_lasso_theta(theta)
        u = np.matrix(np.zeros(beta.size)).T # scaled dual variable
        beta_size = beta.size
        eye_matrix = np.eye(beta_size)

        step_size = init_step_size
        current_value = self.get_value(theta)
        prev_value = current_value

        gd_diff_thres = 0.01
        for i in range(max_iters):
            if i % self.print_iter == 0:
                log.info("ADMM Iter %d, val %f, time %d" % (i, current_value, time.time() - st))

            # UPDATE BETA
            alphas, new_beta, _ = sklearn.linear_model.lasso_path(
                eye_matrix,
                self.get_fused_lasso_theta(theta) - u,
                alphas=[self.penalty_param_fused / beta_size / step_size],
            )
            beta = new_beta[0, :, :]

            # UPDATE THETA
            # pr = SurvivalProblemIshLassoCVXPY(self.feature_generator, self.samples, self.penalty_param_lasso)
            # theta, cvx_val = pr.solve(beta, u, self.D)

            inner_problem = SurvivalProblemIshLassoInner(
                self.feature_vec_sample_pair,
                self.samples,
                self.penalty_param_lasso,
                self.fused_lasso_idx1,
                self.fused_lasso_idx2,
                self.D,
                beta,
                u,
                step_size,
            )
            theta, _ = inner_problem.solve(theta, diff_thres=gd_diff_thres)

            # UPDATE DUAL
            u = u + beta - self.get_fused_lasso_theta(theta)

            current_value = self.get_value(theta)
            diff = prev_value - current_value
            prev_value = current_value
            if diff < diff_thres:
                gd_diff_thres = 0.001
                break
            print "admm", current_value

        log.info("final ADMM iter %d, val %f, time %d" % (i, current_value, time.time() - st))
        return np.array(theta.T)[0], -current_value

    def solve_theta(self):
        # do something
        return 0

class SurvivalProblemIshLassoInner(SurvivalProblemCustom):
    """
    Our own implementation of proximal gradient descent to solve the survival problem
    Objective function: - log likelihood of theta + lasso penalty on theta
    """
    print_iter = 1
    def __init__(self, feature_vec_sample_pair, samples, penalty_param, fused_lasso_idx1, fused_lasso_idx2, D, beta, u, rho):
        """
        @param feat_generator: feature generator
        @param init_theta: where to initialize the gradient descent procedure from
        @param penalty_param: the lasso parameter. should be non-negative
        """
        assert(penalty_param >= 0)

        # self.feature_generator = feat_generator
        self.samples = samples
        self.num_samples = len(self.samples)
        self.feature_vec_sample_pair = feature_vec_sample_pair
        self.penalty_param = penalty_param

        self.fused_lasso_idx1 = fused_lasso_idx1
        self.fused_lasso_idx2 = fused_lasso_idx2
        self.D = D
        self.beta = beta
        self.u = u
        self.rho = rho

    def get_fused_lasso_theta(self, theta):
        return theta[self.fused_lasso_idx1, 0] - theta[self.fused_lasso_idx2, 0]

    def get_value(self, theta):
        """
        @return negative penalized log likelihood
        """
        return -1 * self.get_log_lik(theta) + self.get_value_addon(theta)

    def get_value_addon(self, theta):
        return self.rho/2. * np.power(np.linalg.norm(self.beta - self.D * theta + self.u, ord=2), 2) + self.penalty_param * np.linalg.norm(theta, ord=1)

    def solve(self, init_theta, max_iters=1000, num_threads=1, init_step_size=1, step_size_shrink=0.5, backtrack_alpha = 0.01, diff_thres=0.001, verbose=False):
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
        self.pool = Pool(num_threads)

        st = time.time()
        theta = init_theta
        step_size = init_step_size
        current_value = self.get_value(theta)
        for i in range(max_iters):
            if i % self.print_iter == 0:
                log.info("GD iter %d, val %f, time %d" % (i, current_value, time.time() - st))

            # Calculate gradient of the smooth part
            st = time.time()
            grad_ll = np.matrix(self._get_gradient_log_lik(theta)).T
            print time.time() - st
            st = time.time()
            grad_norm2 = - self.rho * self.D.T * (self.beta - self.get_fused_lasso_theta(theta) + self.u)
            print time.time() - st
            grad = grad_ll + grad_norm2
            potential_theta = theta - step_size * grad

            # Do proximal gradient step
            potential_theta = soft_threshold(potential_theta, step_size * self.penalty_param)
            potential_value = self._get_value_parallel(potential_theta)

            # Do backtracking line search
            expected_decrease = backtrack_alpha * np.power(np.linalg.norm(grad), 2)

            while potential_value >= current_value - step_size * expected_decrease:
                if step_size * expected_decrease < diff_thres:
                    # Stop if difference in objective function is too small
                    break
                step_size *= step_size_shrink
                potential_theta = theta - step_size * grad
                # Do proximal gradient step
                potential_theta = soft_threshold(potential_theta, step_size * self.penalty_param)
                potential_value = self._get_value_parallel(potential_theta)

            if potential_value > current_value:
                # Stop if value is increasing
                break
            else:
                theta = potential_theta
                diff = current_value - potential_value
                current_value = potential_value
                # print "current_value", current_value
                if diff < diff_thres:
                    # Stop if difference in objective function is too small
                    break
        self.pool.close()
        log.info("final GD iter %d, val %f, time %d" % (i, current_value, time.time() - st))

        return theta, -current_value

    def _get_value_parallel(self, theta):
        """
        @return negative penalized log likelihood
        """
        return -1 * self._get_log_lik_parallel(theta) + self.get_value_addon(theta)
