import unittest
import csv
import numpy as np
import scipy as sp

from models import ImputedSequenceMutations, ObservedSequenceMutations
from submotif_feature_generator import SubmotifFeatureGenerator
from mutation_order_gibbs import MutationOrderGibbsSampler
from survival_problem_grad_descent import SurvivalProblemCustom
from common import *

class Survival_Problem_Gradient_Descent_TestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(10)
        cls.motif_len = 3
        cls.BURN_IN = 10
        cls.feat_gen = SubmotifFeatureGenerator(cls.motif_len)
        cls.motif_list = cls.feat_gen.get_motif_list()

    def _compare_grad_calculation(self, theta_num_col):
        """
        Check that the gradient calculation speed up is the same as the old basic gradient calculation
        """
        theta = np.random.rand(self.feat_gen.feature_vec_len, theta_num_col)

        obs = self.feat_gen.create_base_features(
            ObservedSequenceMutations("aggtgggttac", "aggagagttac", self.motif_len)
        )
        sample = ImputedSequenceMutations(obs, obs.mutation_pos_dict.keys())
        theta_mask = get_possible_motifs_to_targets(self.motif_list, theta.shape)
        prob_solver = SurvivalProblemCustom(self.feat_gen, [sample], 1, theta_mask)
        sample_data = prob_solver.precalc_data[0]

        # Basic gradient calculation
        old_grad = self.calculate_grad_slow(theta, sample)

        # Fast gradient calculation
        fast_grad = SurvivalProblemCustom.get_gradient_log_lik_per_sample(
            theta,
            sample_data,
        )
        self.assertTrue(np.allclose(fast_grad, old_grad))

    def test_grad_calculation(self):
        self._compare_grad_calculation(1)
        # self._compare_grad_calculation(NUM_NUCLEOTIDES)

    def _compare_log_likelihood_calculation(self, theta_num_col):
        """
        Check that the log likelihood calculation speed up is the same as the old basic log likelihood calculation
        """
        theta = np.random.rand(self.feat_gen.feature_vec_len, theta_num_col)
        obs = self.feat_gen.create_base_features(
            ObservedSequenceMutations("aggatcgtgatcgagtc", "aaaatcaaaaacgatgc", self.motif_len)
        )
        sample = ImputedSequenceMutations(obs, obs.mutation_pos_dict.keys())
        theta_mask = get_possible_motifs_to_targets(self.motif_list, theta.shape)
        prob_solver = SurvivalProblemCustom(self.feat_gen, [sample], 1, theta_mask)
        sample_data = prob_solver.precalc_data[0]

        feat_mut_steps = self.feat_gen.create_for_mutation_steps(sample)

        # Basic gradient calculation
        old_ll = self.calculate_log_likelihood_slow(theta, sample)
        # Fast log likelihood calculation
        fast_ll = SurvivalProblemCustom.calculate_per_sample_log_lik(theta, sample_data)
        self.assertTrue(np.allclose(fast_ll, old_ll))

    def test_log_likelihood_calculation(self):
        self._compare_log_likelihood_calculation(1)
        # self._compare_log_likelihood_calculation(NUM_NUCLEOTIDES)

    def calculate_grad_slow(self, theta, sample):
        per_target_model = theta.shape[1] == NUM_NUCLEOTIDES

        grad = np.zeros(theta.shape)
        seq_str = sample.obs_seq_mutation.start_seq
        for i in range(len(sample.mutation_order)):
            mutating_pos = sample.mutation_order[i]
            col_idx = 0
            if per_target_model:
                col_idx = NUCLEOTIDE_DICT[sample.obs_seq_mutation.end_seq[mutating_pos]]

            feature_dict = self.feat_gen.create_for_sequence(
                seq_str,
                sample.obs_seq_mutation.left_flank,
                sample.obs_seq_mutation.right_flank,
                set(range(sample.obs_seq_mutation.seq_len)) - set(sample.mutation_order[:i])
            )

            grad[
                feature_dict[mutating_pos],
                col_idx
            ] += 1

            grad_log_sum_exp = np.zeros(theta.shape)

            denom = np.exp(
                [theta[one_feats,i].sum() for one_feats in feature_dict.values() for i in range(theta.shape[1])]
            ).sum()
            for one_feats in feature_dict.values():
                for i in range(theta.shape[1]):
                    val = np.exp(theta[one_feats, i].sum())
                    grad_log_sum_exp[one_feats, i] += val
            grad -= grad_log_sum_exp/denom

            seq_str = mutate_string(
                seq_str,
                mutating_pos,
                sample.obs_seq_mutation.end_seq[mutating_pos]
            )
        return grad

    def calculate_log_likelihood_slow(self, theta, sample):
        per_target_model = theta.shape[1] == NUM_NUCLEOTIDES

        obj = 0
        seq_str = sample.obs_seq_mutation.start_seq
        for i in range(len(sample.mutation_order)):
            mutating_pos = sample.mutation_order[i]
            col_idx = 0
            if per_target_model:
                col_idx = NUCLEOTIDE_DICT[sample.obs_seq_mutation.end_seq[mutating_pos]]

            feature_dict = self.feat_gen.create_for_sequence(
                seq_str,
                sample.obs_seq_mutation.left_flank,
                sample.obs_seq_mutation.right_flank,
                set(range(sample.obs_seq_mutation.seq_len)) - set(sample.mutation_order[:i])
            )
            # vecs_at_mutation_step[i] are the feature vectors of the at-risk group after mutation i
            feature_idx_mutated = feature_dict[mutating_pos]
            col_idx = 0
            if per_target_model:
                col_idx = NUCLEOTIDE_DICT[sample.obs_seq_mutation.end_seq[mutating_pos]]
            log_denom = sp.misc.logsumexp(
                [theta[f,i].sum() for f in feature_dict.values() for i in range(theta.shape[1])]
            )
            obj += theta[feature_idx_mutated, col_idx].sum() - log_denom

            seq_str = mutate_string(
                seq_str,
                mutating_pos,
                sample.obs_seq_mutation.end_seq[mutating_pos]
            )

        return obj
