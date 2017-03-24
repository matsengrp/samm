import time
import unittest
import itertools
import csv
import numpy as np
from cvxpy import *
from common import *

from submotif_feature_generator import SubmotifFeatureGenerator
import solver_wrappers

class Fused_LassoC_TestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(10)
        cls.motif_len = 5
        feat_gen = SubmotifFeatureGenerator(cls.motif_len)
        cls.feature_vec_len = feat_gen.feature_vec_len

        cls.theta_g = np.random.rand(cls.feature_vec_len)

        motif_list = feat_gen.motif_list
        cls.fuse_idx1, cls.fuse_idx2 = cls._get_fuse_indices(motif_list)

    @classmethod
    def _get_fuse_indices(cls, motif_list):
        """
        Get which motifs to fuse together
        """
        fuse_idx1 = []
        fuse_idx2 = []
        for i1, m1 in enumerate(motif_list):
            for i2, m2 in enumerate(motif_list):
                if i1 == i2:
                    continue
                idx_differ = get_idx_differ_by_one_character(m1, m2)
                if idx_differ is None:
                    continue
                else:
                    fuse_idx1.append(i1)
                    fuse_idx2.append(i2)
        return fuse_idx1, fuse_idx2

    def test_fused_lasso(self):
        """
        Test that fused lasso solution from CVXPY is very close to our max-flow solver
        """
        penalty_param = 0.001

        # Run CVXPY
        theta_var = Variable(self.feature_vec_len)
        obj = sum_squares(theta_var - self.theta_g)

        fused_lasso_pen = 0
        for i1, i2 in zip(self.fuse_idx1, self.fuse_idx2):
            fused_lasso_pen += abs(theta_var[i1] - theta_var[i2])

        st = time.time()
        problem = Problem(Minimize(0.5 * obj + penalty_param * fused_lasso_pen))
        problem.solve(verbose=False, solver=SCS, eps=1e-16, max_iters=100000)
        cvx_soln = np.array(theta_var.value.flat)
        print "cvxtime", time.time() - st

        ### Run c code (max-flow solver)
        st = time.time()
        c_soln = np.zeros(self.feature_vec_len, dtype=np.float64)
        solver_wrappers.solve_fused_lasso(
            c_soln,
            np.array(self.theta_g, dtype=np.float64),
            np.array(self.fuse_idx1, dtype=np.intc),
            np.array(self.fuse_idx2, dtype=np.intc),
            penalty_param,
        )
        self.assertTrue(np.allclose(c_soln, cvx_soln))

    def test_sparse_fused_lasso(self):
        """
        Test that sparse fused lasso solution from CVXPY is very close to our max-flow solver
        """
        fuse_penalty_param = 0.001
        lasso_penalty_param = 0.1

        # TEST CVXPY
        theta_var = Variable(self.feature_vec_len)
        obj = sum_squares(theta_var - self.theta_g)

        fused_lasso_pen = 0
        for i1, i2 in zip(self.fuse_idx1, self.fuse_idx2):
            fused_lasso_pen += abs(theta_var[i1] - theta_var[i2])

        st = time.time()
        problem = Problem(Minimize(
            0.5 * obj + fuse_penalty_param * fused_lasso_pen + lasso_penalty_param * norm(theta_var, 1)
        ))
        problem.solve(verbose=False, solver=SCS, eps=1e-16, max_iters=100000)
        cvx_soln = np.array(theta_var.value.flat)
        print "cvxtime", time.time() - st

        st = time.time()
        ### Run c code (max-flow solver)
        c_soln = np.zeros(self.feature_vec_len, dtype=np.float64)
        solver_wrappers.solve_fused_lasso(
            c_soln,
            np.array(self.theta_g, dtype=np.float64),
            np.array(self.fuse_idx1, dtype=np.intc),
            np.array(self.fuse_idx2, dtype=np.intc),
            fuse_penalty_param,
        )
        # Theoretically, we can soft-threshold the solution from the fused lasso
        # to get soln to sparse fused lasso
        sparse_fuse_c_soln = np.multiply(
            np.sign(c_soln),
            np.maximum(
                np.abs(c_soln) - lasso_penalty_param,
                np.zeros(self.feature_vec_len)
            )
        )
        print "c time", time.time() - st
        self.assertTrue(np.allclose(sparse_fuse_c_soln, cvx_soln, rtol=1e-10, atol=1e-14))
