import unittest
import numpy as np

from models import ImputedSequenceMutations, ObservedSequenceMutations
from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator
from survival_problem_cvxpy import SurvivalProblemLassoCVXPY
from survival_problem_lasso import SurvivalProblemLasso
from common import *

class Survival_Problem_TestCase(unittest.TestCase):
    """
    Show that the values from CVXPY and our own impelmentation is the same
    """
    def _test_value_calculation_size(self, theta_num_col):
        np.random.seed(10)
        motif_len = 3
        penalty_param = 0.5

        feat_gen = HierarchicalMotifFeatureGenerator(motif_lens=[motif_len])
        motif_list = feat_gen.motif_list
        theta = np.random.rand(feat_gen.feature_vec_len, theta_num_col)
        theta_mask = feat_gen.get_possible_motifs_to_targets(theta.shape)
        theta[~theta_mask] = -np.inf

        obs = ObservedSequenceMutations("aggtgggttac", "aggagagttac", motif_len)
        feat_gen.add_base_features(obs)
        sample = ImputedSequenceMutations(obs, obs.mutation_pos_dict.keys())
        problem_cvx = SurvivalProblemLassoCVXPY(feat_gen, [sample], penalty_param, theta_mask)
        ll_cvx = problem_cvx.calculate_per_sample_log_lik(theta, sample)
        value_cvx = problem_cvx.get_value(theta)

        feature_mut_steps = feat_gen.create_for_mutation_steps(sample)
        problem_custom = SurvivalProblemLasso(feat_gen, [sample], penalty_param, theta_mask)
        ll_custom = problem_custom.calculate_per_sample_log_lik(np.exp(theta), problem_custom.precalc_data[0])
        value_custom = problem_custom.get_value(theta)
        self.assertTrue(np.isclose(ll_cvx.value, ll_custom))
        self.assertTrue(np.isclose(value_cvx.value, -value_custom))

    def test_value_calculation_size_single(self):
        self._test_value_calculation_size(1)

    @unittest.skip("doesn't work right now")
    def test_value_calculation_size_per_target(self):
        self._test_value_calculation_size(NUM_NUCLEOTIDES)
