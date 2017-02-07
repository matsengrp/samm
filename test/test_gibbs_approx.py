import unittest
import csv
import time
import numpy as np

from mcmc_em import MCMC_EM
from feature_generator import SubmotifFeatureGenerator
from mutation_order_gibbs import MutationOrderGibbsSampler
from survival_problem_lasso import SurvivalProblemLasso
from common import read_gene_seq_csv_data
from constants import *

class Gibbs_TestCase(unittest.TestCase):
    def test_output(self):
        """
        Check if approximations are okay and fast
        """
        feat_generator = SubmotifFeatureGenerator(motif_len=3)
        gene_dict, obs_data = read_gene_seq_csv_data(INPUT_GENES, INPUT_SEQS)

        init_theta = np.random.rand(feat_generator.feature_vec_len, 1)
        theta_mask = np.ones((feat_generator.feature_vec_len, 1), dtype=bool)

        thetas = {}
        traces = []
        for approx in ['none', 'faster', 'fastest']:
            start_time = time.time()
            em_algo = MCMC_EM(
                obs_data,
                feat_generator,
                MutationOrderGibbsSampler,
                SurvivalProblemLasso,
                theta_mask,
                num_threads=1,
                approx=approx,
            )
            theta, trace = em_algo.run(init_theta, max_em_iters=10, penalty_param=0.1)
            thetas[approx] = theta
            traces.append(trace)
            print "approx, time: ", approx, (time.time() - start_time)

        self.assertTrue(np.allclose(thetas['none'], thetas['faster'], rtol=1e-1, atol=1e-1))
        self.assertTrue(np.allclose(thetas['none'], thetas['fastest'], rtol=1e-1, atol=1e-1))

