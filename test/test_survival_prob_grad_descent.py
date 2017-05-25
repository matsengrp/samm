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

        cls.feat_gen_hier = HierarchicalMotifFeatureGenerator(motif_lens=[3,5], left_motif_flank_len_list=[[0,1], [2]])

        obs_seq_mut = ObservedSequenceMutations("agtctggcatcaaagaaagagcgatttag", "aggctcgtattcgctaaaataagcaccag", cls.motif_len)
        cls.mutation_order = [12, 18, 3, 5, 19, 16, 8, 17, 21, 0, 22, 10, 24, 11, 9, 23]

        cls.feat_gen_hier.add_base_features(obs_seq_mut)
        cls.sample_hier = ImputedSequenceMutations(obs_seq_mut, cls.mutation_order)

    def _compare_grad_calculation(self, feat_gen, sample, per_target):
        """
        Check that the gradient calculation speed up is the same as the old basic gradient calculation
        """
        if per_target:
            theta_num_col = NUM_NUCLEOTIDES + 1
        else:
            theta_num_col = 1

        theta = np.random.rand(feat_gen.feature_vec_len, theta_num_col)
        theta_mask = get_possible_motifs_to_targets(feat_gen.motif_list, theta.shape, feat_gen.mutating_pos_list)
        theta[~theta_mask] = -np.inf
        prob_solver = SurvivalProblemCustom(feat_gen, [sample], sample_labels = None, penalty_params=[1], per_target_model=per_target, possible_theta_mask=theta_mask)
        sample_data = prob_solver.precalc_data[0]

        # Basic gradient calculation
        old_grad = self.calculate_grad_slow(theta, feat_gen, sample)

        # Fast gradient calculation
        fast_grad = SurvivalProblemCustom.get_gradient_log_lik_per_sample(
            theta,
            sample_data,
            per_target,
        )
        self.assertTrue(np.allclose(fast_grad, old_grad))

    def test_grad_calculation(self):
        self._compare_grad_calculation(self.feat_gen_hier, self.sample_hier, False)
        self._compare_grad_calculation(self.feat_gen_hier, self.sample_hier, True)

    def _compare_log_likelihood_calculation(self, feat_gen, sample, per_target):
        """
        Check that the log likelihood calculation speed up is the same as the old basic log likelihood calculation
        """
        if per_target:
            theta_num_col = NUM_NUCLEOTIDES + 1
        else:
            theta_num_col = 1

        theta = np.random.rand(feat_gen.feature_vec_len, theta_num_col)
        theta_mask = get_possible_motifs_to_targets(feat_gen.motif_list, theta.shape, feat_gen.mutating_pos_list)
        # theta_mask = get_possible_motifs_to_targets(feat_gen.motif_list, theta.shape)
        theta[~theta_mask] = -np.inf

        prob_solver = SurvivalProblemCustom(feat_gen, [sample], sample_labels = None, penalty_params=[1], per_target_model=per_target, possible_theta_mask=theta_mask)
        sample_data = prob_solver.precalc_data[0]

        feat_mut_steps = feat_gen.create_for_mutation_steps(sample)

        # Basic gradient calculation
        old_ll = self.calculate_log_likelihood_slow(theta, feat_gen, sample)
        # Fast log likelihood calculation
        fast_ll = SurvivalProblemCustom.calculate_per_sample_log_lik(theta, sample_data, per_target)
        self.assertTrue(np.allclose(fast_ll, old_ll))

    def test_log_likelihood_calculation(self):
        self._compare_log_likelihood_calculation(self.feat_gen_hier, self.sample_hier, False)
        self._compare_log_likelihood_calculation(self.feat_gen_hier, self.sample_hier, True)

    def calculate_grad_slow(self, theta, feat_gen, sample):
        per_target_model = theta.shape[1] == NUM_NUCLEOTIDES + 1

        grad = np.zeros(theta.shape)
        seq_str = sample.obs_seq_mutation.start_seq
        denoms = []
        for i in range(len(sample.mutation_order)):
            mutating_pos = sample.mutation_order[i]

            feature_dict = feat_gen.create_for_sequence(
                seq_str,
                sample.obs_seq_mutation.left_flank,
                sample.obs_seq_mutation.right_flank,
                set(range(sample.obs_seq_mutation.seq_len)) - set(sample.mutation_order[:i])
            )
            grad[feature_dict[mutating_pos], 0] += 1
            if per_target_model:
                col_idx = NUCLEOTIDE_DICT[sample.obs_seq_mutation.end_seq[mutating_pos]] + 1
                grad[feature_dict[mutating_pos], col_idx] += 1

            grad_log_sum_exp = np.zeros(theta.shape)
            if per_target_model:
                denom = np.exp([
                    theta[f,0].sum() + theta[f,1:].sum(axis=0) for f in feature_dict.values()
                ]).sum()
            else:
                denom = np.exp([theta[f,0].sum() for f in feature_dict.values()]).sum()

            denoms.append(denom)
            for f in feature_dict.values():
                if per_target_model:
                    grad_log_sum_exp[f, 1:] += np.exp(theta[f, 0].sum() + theta[f, 1:].sum(axis=0))
                    grad_log_sum_exp[f, 0] += np.exp(theta[f, 0].sum() + theta[f, 1:].sum(axis=0)).sum()
                else:
                    grad_log_sum_exp[f, 0] += np.exp(theta[f, 0].sum())
            grad -= grad_log_sum_exp/denom

            seq_str = mutate_string(
                seq_str,
                mutating_pos,
                sample.obs_seq_mutation.end_seq[mutating_pos]
            )
        return grad

    def calculate_log_likelihood_slow(self, theta, feat_gen, sample):
        per_target_model = theta.shape[1] == NUM_NUCLEOTIDES + 1

        obj = 0
        denoms = []
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
            if per_target_model:
                denom = np.exp([
                    theta[f,0].sum() + theta[f,1:].sum(axis=0) for f in feature_dict.values()
                ]).sum()
            else:
                denom = np.exp([theta[f,0].sum() for f in feature_dict.values()]).sum()
            denoms.append(denom)
            log_denom = np.log(denom)

            obj += theta[feature_idx_mutated, 0].sum() - log_denom
            if per_target_model:
                col_idx = NUCLEOTIDE_DICT[sample.obs_seq_mutation.end_seq[mutating_pos]] + 1
                obj += theta[feature_idx_mutated, col_idx].sum()

            seq_str = mutate_string(
                seq_str,
                mutating_pos,
                sample.obs_seq_mutation.end_seq[mutating_pos]
            )
        return obj
