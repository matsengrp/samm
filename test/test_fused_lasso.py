import unittest
import itertools
import csv
import numpy as np
from cvxpy import *
from common import *

from feature_generator import SubmotifFeatureGenerator


class Fused_Lasso_Test:
    def __init__(self):
        np.random.seed(10)
        theta_g = np.random.rand(65)
        print "theta_g", theta_g
        feat_gen = SubmotifFeatureGenerator(3)
        motif_list = feat_gen.get_motif_list()

        with open("test.txt", "w") as f:
            for i, motif in enumerate(motif_list):
                f.write(
                    "%s %d %f\n" % (motif, i, theta_g[i])
                )


        # TEST CVXPY
        theta_var = Variable(feat_gen.feature_vec_len)
        obj = sum_squares(theta_var - theta_g)

        fused_lasso_pen = 0
        for i1, m1 in enumerate(motif_list):
            for i2, m2 in enumerate(motif_list):
                if i1 == i2:
                    continue
                idx_differ = get_idx_differ_by_one_character(m1, m2)
                if idx_differ is None:
                    continue
                else:
                    fused_lasso_pen += abs(theta_var[i1] - theta_var[i2])

        problem = Problem(Minimize(0.5 * obj + 0.001Q * fused_lasso_pen))
        problem.solve(verbose=True)
        assert(problem.status == OPTIMAL)
        print theta_var.value

Fused_Lasso_Test()
