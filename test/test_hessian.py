import unittest
import csv
import numpy as np
import scipy as sp

from models import ImputedSequenceMutations, ObservedSequenceMutations
from submotif_feature_generator import SubmotifFeatureGenerator
from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator
from mutation_order_gibbs import MutationOrderGibbsSampler
from survival_problem_grad_descent import SurvivalProblemCustom
from common import *

class Hessian_TestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(1)
        cls.motif_len = 3
        cls.feat_gen = HierarchicalMotifFeatureGenerator(motif_lens=[3])
        cls.feat_gen_hier = HierarchicalMotifFeatureGenerator(motif_lens=[2,3], left_motif_flank_len_list=[[0,1], [1]])

        cls.obs_seq_mut = ObservedSequenceMutations("agtctggcatcaaagaaagagcgatttag", "aggctcgtattcgctaaaataagcaccag", cls.motif_len)
        cls.mutation_order = [12, 18, 3, 5, 19, 16, 8, 17, 21, 0, 22, 10, 24, 11, 9, 23]

    def _check_hessian_calculation(self, feat_gen, per_target, perturbation = 1e-5):
        if per_target:
            theta_num_col = NUM_NUCLEOTIDES + 1
        else:
            theta_num_col = 1

        theta = np.random.rand(feat_gen.feature_vec_len, theta_num_col)
        possible_theta_mask = get_possible_motifs_to_targets(feat_gen.motif_list, theta.shape, feat_gen.mutating_pos_list)
        theta[~possible_theta_mask] = -np.inf
        zero_theta_mask = np.zeros((feat_gen.feature_vec_len, theta_num_col), dtype=bool)

        feat_gen.add_base_features(self.obs_seq_mut)
        sample = ImputedSequenceMutations(self.obs_seq_mut, self.mutation_order)
        problem = SurvivalProblemCustom(feat_gen, [sample], [1], [0], per_target, possible_theta_mask, zero_theta_mask)

        sample_obs_information, sample_hessian = problem.get_hessian(theta)
        for i in self.obs_seq_mut.feat_matrix_start.nonzero()[1][:5]:
            for j in range(theta.shape[1]):
                perturb_vector = np.zeros(theta.shape)
                perturb_vector[i, j] += perturbation
                sample_gradient1 = problem._get_gradient_log_lik(theta - perturb_vector)
                sample_gradient2 = problem._get_gradient_log_lik(theta + perturb_vector)
                hessian_first_col = (sample_gradient2 - sample_gradient1)/2.0/perturbation
                hessian_first_col = hessian_first_col.reshape((hessian_first_col.size,), order="F")
                assert(np.linalg.norm(hessian_first_col - sample_hessian[:, i + j * theta.shape[0]]) < 1e-9)

    def test_hessian_calculation(self):
        self._check_hessian_calculation(self.feat_gen, False)
        self._check_hessian_calculation(self.feat_gen, True)
        self._check_hessian_calculation(self.feat_gen_hier, False)
        self._check_hessian_calculation(self.feat_gen_hier, True)
