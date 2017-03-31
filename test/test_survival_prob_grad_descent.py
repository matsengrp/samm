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

class Survival_Problem_Gradient_Descent_TestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(10)
        cls.motif_len = 5
        cls.BURN_IN = 10
        cls.feat_gen = SubmotifFeatureGenerator(cls.motif_len)

        cls.feat_gen_hier = HierarchicalMotifFeatureGenerator(motif_lens=[3,5])

        obs_seq_mut = ObservedSequenceMutations("agtctggcatcaaagaaagagcgatttag", "aggctcgtattcgctaaaataagcaccag", cls.motif_len)
        cls.mutation_order = [12, 18, 3, 5, 19, 16, 8, 17, 21, 0, 22, 10, 24, 11, 9, 23]

        obs_seq_mut = ObservedSequenceMutations("agtctggcatcaaagaaagagcgatttag", "agtctggcatcaaataaagtgcgatttag", cls.motif_len)
        cls.mutation_order = [12, 18]

        cls.sample = ImputedSequenceMutations(cls.feat_gen.create_base_features(obs_seq_mut), cls.mutation_order)
        cls.sample_hier = ImputedSequenceMutations(cls.feat_gen_hier.create_base_features(obs_seq_mut), cls.mutation_order)

    def _compare_grad_calculation(self, feat_gen, sample, theta_num_col):
        """
        Check that the gradient calculation speed up is the same as the old basic gradient calculation
        """
        theta = np.random.rand(feat_gen.feature_vec_len, theta_num_col)
        theta_mask = get_possible_motifs_to_targets(feat_gen.motif_list, theta.shape)
        prob_solver = SurvivalProblemCustom(feat_gen, [sample], [1], theta_num_col == NUM_NUCLEOTIDES, theta_mask)
        sample_data = prob_solver.precalc_data[0]

        # Basic gradient calculation
        old_grad = self.calculate_grad_slow(theta, feat_gen, sample)

        # Fast gradient calculation
        # fast_grad = SurvivalProblemCustom.get_gradient_log_lik_per_sample(
        #     np.exp(theta).T,
        #     sample_data,
        # )
        fast_grad = SurvivalProblemCustom.get_gradient_log_lik_per_sample(
            theta,
            sample_data,
        )
        self.assertTrue(np.allclose(fast_grad, old_grad))

    # @unittest.skip("skip")
    def test_grad_calculation(self):
        self._compare_grad_calculation(self.feat_gen_hier, self.sample_hier, 1)
        self._compare_grad_calculation(self.feat_gen_hier, self.sample_hier, NUM_NUCLEOTIDES)

        # self._compare_grad_calculation(self.feat_gen, self.sample, 1)
        # self._compare_grad_calculation(self.feat_gen, self.sample, NUM_NUCLEOTIDES)

    def _compare_log_likelihood_calculation(self, feat_gen, sample, theta_num_col):
        """
        Check that the log likelihood calculation speed up is the same as the old basic log likelihood calculation
        """
        theta = np.random.rand(feat_gen.feature_vec_len, theta_num_col)

        theta_mask = get_possible_motifs_to_targets(feat_gen.motif_list, theta.shape)
        prob_solver = SurvivalProblemCustom(feat_gen, [sample], [1], theta_num_col == NUM_NUCLEOTIDES, theta_mask)
        sample_data = prob_solver.precalc_data[0]

        feat_mut_steps = feat_gen.create_for_mutation_steps(sample)

        # Basic gradient calculation
        old_ll = self.calculate_log_likelihood_slow(theta, feat_gen, sample)
        # Fast log likelihood calculation
        # fast_ll = SurvivalProblemCustom.calculate_per_sample_log_lik(np.exp(theta), sample_data, use_iterative=False)
        fast_ll = SurvivalProblemCustom.calculate_per_sample_log_lik(theta, sample_data)
        self.assertTrue(np.allclose(fast_ll, old_ll))

    def test_log_likelihood_calculation(self):
        self._compare_log_likelihood_calculation(self.feat_gen_hier, self.sample_hier, 1)
        self._compare_log_likelihood_calculation(self.feat_gen_hier, self.sample_hier, NUM_NUCLEOTIDES)

        # self._compare_log_likelihood_calculation(self.feat_gen, self.sample, 1)
        # self._compare_log_likelihood_calculation(self.feat_gen, self.sample, NUM_NUCLEOTIDES)

    def calculate_grad_slow(self, theta, feat_gen, sample):
        per_target_model = theta.shape[1] == NUM_NUCLEOTIDES

        grad = np.zeros(theta.shape)
        seq_str = sample.obs_seq_mutation.start_seq
        denoms = []
        for i in range(len(sample.mutation_order)):
            mutating_pos = sample.mutation_order[i]
            col_idx = 0
            if per_target_model:
                col_idx = NUCLEOTIDE_DICT[sample.obs_seq_mutation.end_seq[mutating_pos]]
            feature_dict = feat_gen.create_for_sequence(
                seq_str,
                sample.obs_seq_mutation.left_flank,
                sample.obs_seq_mutation.right_flank,
                set(range(sample.obs_seq_mutation.seq_len)) - set(sample.mutation_order[:i])
            )

            grad[feature_dict[mutating_pos], col_idx] += 1

            grad_log_sum_exp = np.zeros(theta.shape)
            denom = np.exp([theta[f,:].sum(axis=0) for f in feature_dict.values()]).sum()
            denoms.append(denom)
            for f in feature_dict.values():
                grad_log_sum_exp[f, :] += np.exp(theta[f, :].sum(axis=0))
            grad -= grad_log_sum_exp/denom

            seq_str = mutate_string(
                seq_str,
                mutating_pos,
                sample.obs_seq_mutation.end_seq[mutating_pos]
            )
        return grad

    def calculate_log_likelihood_slow(self, theta, feat_gen, sample):
        per_target_model = theta.shape[1] == NUM_NUCLEOTIDES

        obj = 0
        seq_str = sample.obs_seq_mutation.start_seq
        for i in range(len(sample.mutation_order)):
            mutating_pos = sample.mutation_order[i]

            feature_dict = feat_gen.create_for_sequence(
                seq_str,
                sample.obs_seq_mutation.left_flank,
                sample.obs_seq_mutation.right_flank,
                set(range(sample.obs_seq_mutation.seq_len)) - set(sample.mutation_order[:i])
            )
            # vecs_at_mutation_step[i] are the feature vectors of the at-risk group after mutation i
            feature_idx_mutated = feature_dict[mutating_pos]
            denom = np.exp([theta[f,:].sum(axis=0) for f in feature_dict.values()]).sum()
            log_denom = np.log(denom)

            col_idx = 0
            if per_target_model:
                col_idx = NUCLEOTIDE_DICT[sample.obs_seq_mutation.end_seq[mutating_pos]]
            obj += theta[feature_idx_mutated, col_idx].sum() - log_denom

            seq_str = mutate_string(
                seq_str,
                mutating_pos,
                sample.obs_seq_mutation.end_seq[mutating_pos]
            )
        return obj
