import unittest
import csv

from mcmc_em import MCMC_EM
from feature_generator import SubmotifFeatureGenerator
from mutation_order_gibbs import MutationOrderGibbsSampler
from survival_problem_grad_descent import SurvivalProblemGradientDescent
from common import read_gene_seq_csv_data
from constants import *

class MCMC_EM_TestCase(unittest.TestCase):
    def test_mini(self):
        """
        Check if MCMC EM will run to completion
        """
        feat_generator = SubmotifFeatureGenerator(submotif_len=3)
        gene_dict, obs_data = read_gene_seq_csv_data(INPUT_GENES, INPUT_SEQS)
        em_algo = MCMC_EM(
            obs_data,
            feat_generator,
            MutationOrderGibbsSampler,
            SurvivalProblemGradientDescent,
            num_threads=1,
        )
        em_algo.run(max_em_iters=1)
