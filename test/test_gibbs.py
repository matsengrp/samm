import unittest
import itertools
import csv
import numpy as np
from scipy.stats import spearmanr
from collections import Counter

from common import *
from models import ObservedSequenceMutations
from survival_model_simulator import SurvivalModelSimulatorSingleColumn
from survival_model_simulator import SurvivalModelSimulatorMultiColumn
from submotif_feature_generator import SubmotifFeatureGenerator
from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator
from mutation_order_gibbs import MutationOrderGibbsSampler, GibbsStepInfo

class Gibbs_TestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(10)
        cls.motif_len = 3
        cls.BURN_IN = 10
        cls.feat_gen = HierarchicalMotifFeatureGenerator(motif_lens=[3], left_motif_flank_len_list=[[1]])
        cls.feat_gen_hier = HierarchicalMotifFeatureGenerator(motif_lens=[1,3], left_motif_flank_len_list=[[0],[1]])
        cls.obs = ObservedSequenceMutations("attcaaatgatatac", "ataaatagggtttac", cls.motif_len, left_flank_len=1, right_flank_len=1)

        cls.feat_gen_off = HierarchicalMotifFeatureGenerator(motif_lens=[3], left_motif_flank_len_list=[[0,1,2]])
        cls.obs_off = ObservedSequenceMutations("attcaaatgatatac", "ataaatagggtttac", cls.motif_len, left_flank_len=2, right_flank_len=2)

    def _test_compute_log_probs(self, feat_gen, per_target_model, obs_seq_m):
        if per_target_model:
            theta = np.random.rand(feat_gen.feature_vec_len, NUM_NUCLEOTIDES + 1)
            possible_motif_mask = get_possible_motifs_to_targets(feat_gen.motif_list,
                    theta.shape,
                    feat_gen.mutating_pos_list)
            theta[~possible_motif_mask] = -np.inf
        else:
            theta = np.random.rand(feat_gen.feature_vec_len, 1) * 2
        feat_gen.add_base_features(obs_seq_m)
        sampler = MutationOrderGibbsSampler(theta, feat_gen, obs_seq_m)

        order = obs_seq_m.mutation_pos_dict.keys()

        # This calculates denominators efficiently using deltas
        feat_mut_steps, log_numerators, denominators = sampler._compute_log_probs_from_scratch(order)

        self.assertEqual(len(log_numerators), len(order))
        self.assertEqual(len(denominators), len(order))
        seq_str = obs_seq_m.start_seq
        for i in range(len(order)):
            mutating_pos = order[i]
            log_num = theta[feat_mut_steps[i].mutating_pos_feats, 0].sum()
            if per_target_model:
                col_idx = NUCLEOTIDE_DICT[obs_seq_m.end_seq[mutating_pos]] + 1
                log_num += theta[feat_mut_steps[i].mutating_pos_feats, col_idx].sum()

            feature_dict = feat_gen.create_for_sequence(
                seq_str,
                obs_seq_m.left_flank,
                obs_seq_m.right_flank,
                set(range(obs_seq_m.seq_len)) - set(order[:i])
            )
            # Calculates denominators from scratch - sum(exp(psi * theta))
            if not per_target_model:
                denom = np.exp([
                    theta[feat_idx, 0].sum() for feat_idx in feature_dict.values()
                ]).sum()
            else:
                denom = np.exp([
                    theta[feat_idx, 0].sum() + theta[feat_idx, 1:].sum(axis=0) for feat_idx in feature_dict.values()
                ]).sum()
            self.assertEqual(log_num, log_numerators[i])
            self.assertTrue(np.isclose(denom, denominators[i]))

            seq_str = mutate_string(
                seq_str,
                order[i],
                obs_seq_m.end_seq[order[i]]
            )

    def test_compute_log_probs(self):
        for per_target_model in [False, True]:
            self._test_compute_log_probs(self.feat_gen, per_target_model, self.obs)
            self._test_compute_log_probs(self.feat_gen_hier, per_target_model, self.obs)
            self._test_compute_log_probs(self.feat_gen_off, per_target_model, self.obs_off)

    def _test_compute_log_probs_with_reference(self, feat_gen, per_target_model, obs_seq_m):
        feat_gen.add_base_features(obs_seq_m)
        if per_target_model:
            num_cols = NUM_NUCLEOTIDES + 1
        else:
            num_cols = 1
        theta = np.random.rand(feat_gen.feature_vec_len, num_cols) * 2
        sampler = MutationOrderGibbsSampler(theta, feat_gen, obs_seq_m)

        prev_order = obs_seq_m.mutation_pos_dict.keys()
        curr_order = prev_order[:2] + [prev_order[3], prev_order[2]] + prev_order[4:]

        prev_feat_mutation_steps, prev_log_numerators, prev_denominators = sampler._compute_log_probs_from_scratch(
            prev_order,
        )

        _, curr_log_numerators, curr_denominators = sampler._compute_log_probs_from_scratch(
            curr_order,
        )

        gibbs_step_info = GibbsStepInfo(
            prev_order,
            prev_log_numerators,
            prev_denominators,
        )
        _, fast_log_numerators, fast_denominators = sampler._compute_log_probs_with_reference(
            curr_order,
            gibbs_step_info,
            update_step_start=2,
        )
        self.assertTrue(np.allclose(curr_denominators, fast_denominators))
        self.assertTrue(np.allclose(curr_log_numerators, fast_log_numerators))

    def test_compute_log_probs_with_reference(self):
        for per_target_model in [False, True]:
            self._test_compute_log_probs_with_reference(self.feat_gen, per_target_model, self.obs)
            self._test_compute_log_probs_with_reference(self.feat_gen_hier, per_target_model, self.obs)
            self._test_compute_log_probs_with_reference(self.feat_gen_off, per_target_model, self.obs_off)

    def _test_joint_distribution(self, feat_gen, theta):
        """
        Check that the distribution of mutation orders is similar when we generate mutation orders directly
        from the survival model vs. when we generate mutation orders given the mutation positions from the
        gibbs sampler
        """
        START_SEQ = "attcgc" # MUST BE LESS THAN TEN, Includes flanks!
        BURN_IN = 15
        CENSORING_TIME = 2.0
        LAMBDA0 = 0.1
        NUM_TOP_COMMON = 20
        NUM_OBS_SAMPLES=8000

        per_target_model = theta.shape[1] == NUM_NUCLEOTIDES + 1
        if not per_target_model:
            probability_matrix = np.ones((feat_gen.feature_vec_len, NUM_NUCLEOTIDES))/3.0
            possible_motif_mask = get_possible_motifs_to_targets(feat_gen.motif_list,
                    (feat_gen.feature_vec_len, NUM_NUCLEOTIDES),
                    feat_gen.mutating_pos_list)
            probability_matrix[~possible_motif_mask] = 0
            surv_simulator = SurvivalModelSimulatorSingleColumn(theta, probability_matrix, feat_gen, lambda0=LAMBDA0)
        else:
            surv_simulator = SurvivalModelSimulatorMultiColumn(theta[:,0:1] + theta[:, 1:], feat_gen, lambda0=LAMBDA0)

        # Simulate some data from the same starting sequence
        # Get the distribution of mutation orders from our survival model
        full_seq_muts = [surv_simulator.simulate(START_SEQ, censoring_time=CENSORING_TIME) for i in range(NUM_OBS_SAMPLES)]
        # We make the mutation orders strings so easy to process
        true_order_distr = ["".join(map(str,m.get_mutation_order())) for m in full_seq_muts]
        obs_seq_mutations = []
        for m in full_seq_muts:
            obs = ObservedSequenceMutations(
                m.left_flank + m.start_seq + m.right_flank,
                m.left_flank + m.end_seq + m.right_flank,
                motif_len=self.motif_len,
                left_flank_len=feat_gen.max_left_motif_flank_len,
                right_flank_len=feat_gen.max_right_motif_flank_len,
            )
            feat_gen.add_base_features(obs)
            obs_seq_mutations.append(obs)

        # Now get the distribution of orders from our gibbs sampler (so sample mutation order
        # given known mutation positions)
        gibbs_order = []
        for i, obs_seq_m in enumerate(obs_seq_mutations):
            gibbs_sampler = MutationOrderGibbsSampler(theta, feat_gen, obs_seq_m)
            gibbs_samples = gibbs_sampler.run(obs_seq_m.mutation_pos_dict.keys(), BURN_IN, 1)
            order_sample = gibbs_samples.samples[0].mutation_order
            order_sample = map(str, order_sample)
            # We make the mutation orders strings so easy to process
            gibbs_order.append("".join(order_sample))

        # Now count the number of times each mutation order occurs
        true_counter = Counter(true_order_distr)
        gibbs_counter = Counter(gibbs_order)
        all_orders = set(true_counter.keys()).union(set(gibbs_counter.keys()))
        true_counter_order_vec = [max(true_counter.get(o), 0) for o in all_orders]
        gibbs_counter_order_vec = [max(gibbs_counter.get(o), 0) for o in all_orders]

        # Calculate the spearman correlation. Should be high!
        rho, pval = spearmanr(true_counter_order_vec, gibbs_counter_order_vec)
        print "rho, pval", rho, pval, per_target_model

        for t, g in zip(true_counter.most_common(NUM_TOP_COMMON), gibbs_counter.most_common(NUM_TOP_COMMON)):
            print "%s (%d) \t %s (%d)" % (t[0], t[1], g[0], g[1])
        return rho, pval

    def test_joint_distribution_simple(self):
        """
        Test the joint distributions match for a single column theta (not a per-target-nucleotide model)
        """
        theta = np.random.rand(self.feat_gen.feature_vec_len, 1) * 2
        rho, pval = self._test_joint_distribution(self.feat_gen, theta)
        self.assertTrue(rho > 0.95)
        self.assertTrue(pval < 1e-33)

        theta = np.random.rand(self.feat_gen_hier.feature_vec_len, 1)
        rho, pval = self._test_joint_distribution(self.feat_gen_hier, theta)
        self.assertTrue(rho > 0.95)
        self.assertTrue(pval < 1e-33)

    def test_joint_distribution_per_target_model(self):
        """
        Test the joint distributions match for a single column theta (not a per-target-nucleotide model)
        """
        def _make_multi_theta(feat_gen):
            # This generates a theta with random entries
            multi_theta = np.random.rand(feat_gen.feature_vec_len, NUM_NUCLEOTIDES + 1)
            theta_mask = get_possible_motifs_to_targets(feat_gen.motif_list,
                    multi_theta.shape,
                    feat_gen.mutating_pos_list)
            multi_theta[~theta_mask] = -np.inf
            return multi_theta

        multi_theta = _make_multi_theta(self.feat_gen)/2
        rho, pval = self._test_joint_distribution(self.feat_gen, multi_theta)
        self.assertTrue(rho > 0.90)
        self.assertTrue(pval < 1e-25)

        multi_theta = _make_multi_theta(self.feat_gen_hier)/2
        rho, pval = self._test_joint_distribution(self.feat_gen_hier, multi_theta)
        self.assertTrue(rho > 0.97)
        self.assertTrue(pval < 1e-40)
