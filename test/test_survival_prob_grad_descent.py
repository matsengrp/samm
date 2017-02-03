import unittest
import csv
import numpy as np
import scipy as sp

from models import ImputedSequenceMutations, ObservedSequenceMutations
from feature_generator import SubmotifFeatureGenerator
from mutation_order_gibbs import MutationOrderGibbsSampler
from survival_problem_grad_descent import SurvivalProblemCustom
from common import NUM_NUCLEOTIDES, NUCLEOTIDE_DICT

class Survival_Problem_Gradient_Descent_TestCase(unittest.TestCase):
    def _compare_grad_calculation(self, theta_num_col):
        """
        Check that the gradient calculation speed up is the same as the old basic gradient calculation
        """
        np.random.seed(10)
        motif_len = 3

        feat_gen = SubmotifFeatureGenerator(motif_len)

        theta = np.random.rand(feat_gen.feature_vec_len, theta_num_col)
        obs = ObservedSequenceMutations("ggtgggtta", "ggagagtta")
        sample = ImputedSequenceMutations(obs, obs.mutation_pos_dict.keys())
        feature_vecs = feat_gen.create_for_mutation_steps(sample)[0]

        # Basic gradient calculation
        old_grad = self.calculate_grad_slow(theta, sample, feature_vecs)
        # Fast gradient calculation
        fast_grad = SurvivalProblemCustom.get_gradient_log_lik_per_sample(theta, sample, feature_vecs, motif_len)
        self.assertTrue(np.allclose(fast_grad, old_grad))

    def test_grad_calculation(self):
        self._compare_grad_calculation(1)
        self._compare_grad_calculation(NUM_NUCLEOTIDES)

    def _compare_log_likelihood_calculation(self, theta_num_col):
        """
        Check that the log likelihood calculation speed up is the same as the old basic log likelihood calculation
        """
        np.random.seed(10)
        motif_len = 3

        feat_gen = SubmotifFeatureGenerator(motif_len)

        theta = np.random.rand(feat_gen.feature_vec_len, theta_num_col)
        obs = ObservedSequenceMutations("ggatcgtgatcgagt", "aaatcaaaaacgatg")
        sample = ImputedSequenceMutations(obs, obs.mutation_pos_dict.keys())
        feature_vecs = feat_gen.create_for_mutation_steps(sample)[0]

        # Basic gradient calculation
        old_ll = self.calculate_log_likelihood_slow(theta, sample, feature_vecs)
        # Fast log likelihood calculation
        fast_ll = SurvivalProblemCustom.calculate_per_sample_log_lik(theta, sample, feature_vecs, motif_len)
        self.assertTrue(np.allclose(fast_ll, old_ll))

    def test_log_likelihood_calculation(self):
        self._compare_log_likelihood_calculation(1)
        self._compare_log_likelihood_calculation(NUM_NUCLEOTIDES)

    def calculate_grad_slow(self, theta, sample, feature_vecs):
        per_target_model = theta.shape[1] == NUM_NUCLEOTIDES
        grad = np.zeros(theta.shape)
        for mutating_pos, vecs_at_mutation_step in zip(sample.mutation_order, feature_vecs):
            col_idx = 0
            if per_target_model:
                col_idx = NUCLEOTIDE_DICT[sample.obs_seq_mutation.end_seq[mutating_pos]]

            grad[
                vecs_at_mutation_step[mutating_pos],
                col_idx
            ] += 1

            grad_log_sum_exp = np.zeros(theta.shape)
            denom = np.exp(
                [theta[one_feats,i].sum() for one_feats in vecs_at_mutation_step.values() for i in range(theta.shape[1])]
            ).sum()
            for one_feats in vecs_at_mutation_step.values():
                for i in range(theta.shape[1]):
                    val = np.exp(theta[one_feats, i].sum())
                    grad_log_sum_exp[one_feats, i] += val
            grad -= grad_log_sum_exp/denom
        return grad

    def calculate_log_likelihood_slow(self, theta, sample, feature_vecs):
        obj = 0
        for mutating_pos, vecs_at_mutation_step in zip(sample.mutation_order, feature_vecs):
            # vecs_at_mutation_step[i] are the feature vectors of the at-risk group after mutation i
            feature_vec_mutated = vecs_at_mutation_step[mutating_pos]
            obj += theta[feature_vec_mutated].sum() - sp.misc.logsumexp(
                [theta[f].sum() for f in vecs_at_mutation_step.values()]
            )
        return obj
