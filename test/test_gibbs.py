import unittest
import itertools
import csv
import numpy as np
from scipy.stats import spearmanr
from collections import Counter

from common import *
from models import ObservedSequenceMutations
from survival_model_simulator import SurvivalModelSimulator
from submotif_feature_generator import SubmotifFeatureGenerator
from mutation_order_gibbs import MutationOrderGibbsSampler, GibbsStepInfo

class MCMC_EM_TestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(10)
        cls.motif_len = 3
        cls.BURN_IN = 10
        cls.feat_gen = SubmotifFeatureGenerator(cls.motif_len)
        motif_list = cls.feat_gen.get_motif_list()
        motif_list_len = len(motif_list)
        # NOTE: THIS IS A SMALL THETA
        cls.big_theta = np.repeat(np.random.rand(cls.feat_gen.feature_vec_len, 1), NUM_NUCLEOTIDES, axis=1)
        cls.big_theta[:motif_list_len,] = cls.big_theta[:motif_list_len,] - np.log(3)
        cls.big_theta[motif_list_len,] = cls.big_theta[motif_list_len,] - np.log(4)
        theta_mask = get_possible_motifs_to_targets(cls.feat_gen.get_motif_list(), cls.big_theta.shape)
        cls.big_theta[~theta_mask] = -np.inf
        cls.theta = np.matrix(np.max(cls.big_theta, axis=1)).T
        cls.obs_seq_m = cls.feat_gen.create_base_features(ObservedSequenceMutations("ttcgtata", "taagttat"))
        cls.gibbs_sampler = MutationOrderGibbsSampler(cls.theta, cls.feat_gen, cls.obs_seq_m)

    def test_update_log_prob_for_shuffle(self):
        prev_order = self.obs_seq_m.mutation_pos_dict.keys()
        curr_order = prev_order[:2] + [prev_order[3], prev_order[2]] + prev_order[4:]

        _, prev_log_probs, prev_feature_vec_theta_sums = self.gibbs_sampler._compute_log_probs(prev_order)
        _, slow_log_probs, feature_vec_theta_sums = self.gibbs_sampler._compute_log_probs(curr_order)
        fast_log_probs = self.gibbs_sampler._update_log_prob_from_shuffle(2, prev_log_probs, curr_order, prev_order, feature_vec_theta_sums, prev_feature_vec_theta_sums)

        self.assertTrue(np.allclose(slow_log_probs, fast_log_probs))

    def test_update_log_prob_for_positions(self):
        curr_order = self.obs_seq_m.mutation_pos_dict.keys()
        curr_feat_mut_steps, curr_log_probs, curr_feature_vec_theta_sums = self.gibbs_sampler._compute_log_probs(curr_order)

        new_order = [curr_order[0], curr_order[-1]] + curr_order[1:-1]
        feat_mut_steps_slow, multinomial_sequence_slow, feature_vec_theta_sums_slow = self.gibbs_sampler._compute_log_probs(new_order)
        feat_mut_steps, multinomial_sequence, feature_vec_theta_sums = self.gibbs_sampler._compute_log_probs(
            new_order,
            GibbsStepInfo(curr_order, curr_feat_mut_steps, curr_feature_vec_theta_sums, curr_log_probs),
            update_positions=range(1, len(new_order)),
        )
        self.assertEqual(feat_mut_steps_slow.intermediate_seqs, feat_mut_steps.intermediate_seqs)
        self.assertEqual(feat_mut_steps_slow.feature_vec_dicts, feat_mut_steps.feature_vec_dicts)
        self.assertTrue(np.allclose(multinomial_sequence, multinomial_sequence_slow))
        self.assertEqual(feature_vec_theta_sums, feature_vec_theta_sums_slow)

    def _test_joint_distribution(self, theta):
        """
        Check that the distribution of mutation orders is similar when we generate mutation orders directly
        from the survival model vs. when we generate mutation orders given the mutation positions from the
        gibbs sampler
        """
        START_SEQ = "ttcg" # MUST BE LESS THAN TEN
        NUM_OBS_SAMPLES = 5000
        BURN_IN = 15
        CENSORING_TIME = 2.0
        LAMBDA0 = 0.1
        NUM_TOP_COMMON = 20

        surv_simulator = SurvivalModelSimulator(theta, self.feat_gen, lambda0=LAMBDA0)

        # Simulate some data from the same starting sequence
        # Get the distribution of mutation orders from our survival model
        full_seq_muts = [surv_simulator.simulate(START_SEQ, censoring_time=CENSORING_TIME) for i in range(NUM_OBS_SAMPLES)]
        # We make the mutation orders strings so easy to process
        true_order_distr = ["".join(map(str,m.get_mutation_order())) for m in full_seq_muts]
        obs_seq_mutations = [
            self.feat_gen.create_base_features(ObservedSequenceMutations(START_SEQ, m.obs_seq_mutation.end_seq)) for m in full_seq_muts
        ]

        # Now get the distribution of orders from our gibbs sampler (so sample mutation order
        # given known mutation positions)
        gibbs_order = []
        for i, obs_seq_m in enumerate(obs_seq_mutations):
            gibbs_sampler = MutationOrderGibbsSampler(theta, self.feat_gen, obs_seq_m)
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
        print "rho, pval", rho, pval

        for t, g in zip(true_counter.most_common(NUM_TOP_COMMON), gibbs_counter.most_common(NUM_TOP_COMMON)):
            print "%s (%d) \t %s (%d)" % (t[0], t[1], g[0], g[1])
        return rho, pval

    def test_joint_distribution_simple(self):
        """
        Test the joint distributions match for a single column theta (not a per-target-nucleotide model)
        """
        rho, pval = self._test_joint_distribution(self.big_theta)
        self.assertTrue(rho > 0.93)
        self.assertTrue(pval < 1e-23)

    def test_joint_distribution_per_target_model(self):
        """
        Test the joint distributions match for a single column theta (not a per-target-nucleotide model)
        """
        theta = np.random.rand(self.feat_gen.feature_vec_len, NUM_NUCLEOTIDES) * 4
        theta_mask = get_possible_motifs_to_targets(self.feat_gen.get_motif_list(), theta.shape)
        theta[~theta_mask] = -np.inf

        rho, pval = self._test_joint_distribution(theta)

        self.assertTrue(rho > 0.98)
        self.assertTrue(pval < 1e-37)
