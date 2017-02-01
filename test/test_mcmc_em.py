import unittest
import csv

from mcmc_em import MCMC_EM
from feature_generator import SubmotifFeatureGenerator
from mutation_order_gibbs import MutationOrderGibbsSampler
from survival_problem_cvxpy import SurvivalProblemLassoCVXPY
from survival_problem_cvxpy import SurvivalProblemFusedLassoCVXPY
from survival_problem_lasso import SurvivalProblemLasso
from survival_problem_fused_lasso import SurvivalProblemFusedLasso
from common import read_gene_seq_csv_data
from constants import *

class MCMC_EM_TestCase(unittest.TestCase):
    def test_mini(self):
        """
        Check if MCMC EM will run to completion
        """
        feat_generator = SubmotifFeatureGenerator(motif_len=3)
        gene_dict, obs_data = read_gene_seq_csv_data(INPUT_GENES, INPUT_SEQS)

        # check SurvivalProblemLasso
        em_algo = MCMC_EM(
            obs_data,
            feat_generator,
            MutationOrderGibbsSampler,
            SurvivalProblemLasso,
            num_threads=1,
        )
        em_algo.run(max_em_iters=1)

        # check SurvivalProblemFusedLasso
        em_algo = MCMC_EM(
            obs_data,
            feat_generator,
            MutationOrderGibbsSampler,
            SurvivalProblemFusedLasso,
            num_threads=1,
        )
        em_algo.run(max_em_iters=1)

        # Check SurvivalProblemLassoCVXPY
        em_algo = MCMC_EM(
            obs_data,
            feat_generator,
            MutationOrderGibbsSampler,
            SurvivalProblemLassoCVXPY,
            num_threads=1,
        )
        em_algo.run(max_em_iters=1)

        # Check SurvivalProblemFusedLassoCVXPY
        em_algo = MCMC_EM(
            obs_data,
            feat_generator,
            MutationOrderGibbsSampler,
            SurvivalProblemFusedLassoCVXPY,
            num_threads=1,
        )
        em_algo.run(max_em_iters=1)
