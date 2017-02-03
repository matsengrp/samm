import unittest
import csv
import numpy as np
import scipy as sp

from models import ImputedSequenceMutations, ObservedSequenceMutations
from feature_generator import SubmotifFeatureGenerator
from mutation_order_gibbs import MutationOrderGibbsSampler
from survival_problem_grad_descent import SurvivalProblemCustom

class Survival_Problm_Gradient_Descent_TestCase(unittest.TestCase):
    def test_grad_calculation(self):
        """
        Check that the gradient calculation speed up is the same as the old basic gradient calculation
        """
        np.random.seed(10)
        motif_len = 3

        feat_gen = SubmotifFeatureGenerator(motif_len)

        theta = np.random.rand(feat_gen.feature_vec_len)
        theta[feat_gen.feature_vec_len - 1] = -1
        obs = ObservedSequenceMutations("ggtgggtta", "ggagagtta")
        sample = ImputedSequenceMutations(obs, [4,2])
        feature_vecs = feat_gen.create_for_mutation_steps(sample)[0]

        # Basic gradient calculation
        old_grad = self.calculate_grad_slow(theta, sample, feature_vecs)
        # Fast gradient calculation
        fast_grad = SurvivalProblemCustom.get_gradient_log_lik_per_sample(theta, sample, feature_vecs, motif_len)
        self.assertTrue(np.allclose(fast_grad, old_grad))

    def test_log_likelihood_calculation(self):
        """
        Check that the log likelihood calculation speed up is the same as the old basic log likelihood calculation
        """
        np.random.seed(10)
        motif_len = 3

        feat_gen = SubmotifFeatureGenerator(motif_len)

        theta = np.random.rand(feat_gen.feature_vec_len)
        obs = ObservedSequenceMutations("ggatcgtgatcgagt", "aaatcaaaaacgatg")
        sample = ImputedSequenceMutations(obs, obs.mutation_pos_dict.keys())
        feature_vecs = feat_gen.create_for_mutation_steps(sample)[0]

        # Basic gradient calculation
        old_ll = self.calculate_log_likelihood_slow(theta, sample, feature_vecs)
        # Fast log likelihood calculation
        fast_ll = SurvivalProblemCustom.calculate_per_sample_log_lik(theta, sample, feature_vecs, motif_len)
        self.assertTrue(np.allclose(fast_ll, old_ll))

    def calculate_grad_slow(self, theta, sample, feature_vecs):
        grad = np.zeros(theta.size)
        for mutating_pos, vecs_at_mutation_step in zip(sample.mutation_order, feature_vecs):
            grad[vecs_at_mutation_step[mutating_pos]] += 1
            grad_log_sum_exp = np.zeros(theta.size)
            denom = np.exp([theta[one_feats].sum() for one_feats in vecs_at_mutation_step.values()]).sum()
            for one_feats in vecs_at_mutation_step.values():
                val = np.exp(theta[one_feats].sum())
                grad_log_sum_exp[one_feats] += val
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
