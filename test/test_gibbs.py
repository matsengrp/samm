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

class Gibbs_TestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(10)
        cls.motif_len = 3
        cls.BURN_IN = 10
        cls.feat_gen = SubmotifFeatureGenerator(cls.motif_len)
        motif_list = cls.feat_gen.get_motif_list()
        motif_list_len = len(motif_list)
        # This generates a theta where columns are repeated
        cls.repeat_theta = np.repeat(np.random.rand(cls.feat_gen.feature_vec_len, 1), NUM_NUCLEOTIDES, axis=1)
        cls.repeat_theta[:motif_list_len,] = cls.repeat_theta[:motif_list_len,]
        theta_mask = get_possible_motifs_to_targets(cls.feat_gen.get_motif_list(), cls.repeat_theta.shape)
        cls.repeat_theta[~theta_mask] = -np.inf
        cls.theta = np.matrix(np.max(cls.repeat_theta, axis=1)).T
        cls.obs_seq_m = cls.feat_gen.create_base_features(ObservedSequenceMutations("attcgtatac", "ataagttatc", cls.motif_len))
        cls.gibbs_sampler = MutationOrderGibbsSampler(cls.theta, cls.feat_gen, cls.obs_seq_m)

    def test_compute_log_probs(self):
        order = self.obs_seq_m.mutation_pos_dict.keys()

        # This calculates denominators efficiently using deltas
        feat_mut_steps, log_numerators, denominators = self.gibbs_sampler._compute_log_probs_from_scratch(
            order,
        )

        self.assertEqual(len(log_numerators), len(order))
        self.assertEqual(len(denominators), len(order))
        seq_str = self.obs_seq_m.start_seq
        for i in range(len(order)):
            log_num = self.theta[feat_mut_steps[i].mutating_pos_feat]
            feature_dict = self.feat_gen.create_for_sequence(
                seq_str,
                self.obs_seq_m.left_flank,
                self.obs_seq_m.right_flank,
                set(range(self.obs_seq_m.seq_len)) - set(order[:i])
            )
            # Calculates denominators from scratch - sum(exp(psi * theta))
            denom = np.exp([
                self.theta[feat_idx] for feat_idx in feature_dict.values()
            ]).sum()
            self.assertEqual(log_num, log_numerators[i])
            self.assertTrue(np.isclose(denom, denominators[i]))

            seq_str = mutate_string(
                seq_str,
                order[i],
                self.obs_seq_m.end_seq[order[i]]
            )

    def test_compute_log_probs_with_reference(self):
        prev_order = self.obs_seq_m.mutation_pos_dict.keys()
        curr_order = prev_order[:2] + [prev_order[3], prev_order[2]] + prev_order[4:]

        prev_feat_mutation_steps, prev_log_numerators, prev_denominators = self.gibbs_sampler._compute_log_probs_from_scratch(
            prev_order,
        )

        _, curr_log_numerators, curr_denominators = self.gibbs_sampler._compute_log_probs_from_scratch(
            curr_order,
        )

        gibbs_step_info = GibbsStepInfo(
            prev_order,
            prev_log_numerators,
            prev_denominators,
        )
        _, fast_log_numerators, fast_denominators = self.gibbs_sampler._compute_log_probs_with_reference(
            curr_order,
            gibbs_step_info,
            update_step_start=2,
        )
        self.assertTrue(np.allclose(curr_denominators, fast_denominators))
        self.assertTrue(np.allclose(curr_log_numerators, fast_log_numerators))

    def _test_joint_distribution(self, simulator_theta, gibbs_theta):
        """
        Check that the distribution of mutation orders is similar when we generate mutation orders directly
        from the survival model vs. when we generate mutation orders given the mutation positions from the
        gibbs sampler
        """
        START_SEQ = "attcgc" # MUST BE LESS THAN TEN
        NUM_OBS_SAMPLES = 5500
        BURN_IN = 15
        CENSORING_TIME = 2.0
        LAMBDA0 = 0.1
        NUM_TOP_COMMON = 20

        surv_simulator = SurvivalModelSimulator(simulator_theta, self.feat_gen, lambda0=LAMBDA0)
        # Simulate some data from the same starting sequence
        # Get the distribution of mutation orders from our survival model
        full_seq_muts = [surv_simulator.simulate(START_SEQ, censoring_time=CENSORING_TIME) for i in range(NUM_OBS_SAMPLES)]
        # We make the mutation orders strings so easy to process
        true_order_distr = ["".join(map(str,m.get_mutation_order())) for m in full_seq_muts]
        obs_seq_mutations = [
            self.feat_gen.create_base_features(
                ObservedSequenceMutations(
                    m.left_flank + m.start_seq + m.right_flank,
                    m.left_flank + m.end_seq + m.right_flank,
                    self.motif_len,
                )
            ) for m in full_seq_muts
        ]

        # Now get the distribution of orders from our gibbs sampler (so sample mutation order
        # given known mutation positions)
        gibbs_order = []
        for i, obs_seq_m in enumerate(obs_seq_mutations):
            gibbs_sampler = MutationOrderGibbsSampler(gibbs_theta, self.feat_gen, obs_seq_m)
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
        rho, pval = self._test_joint_distribution(self.repeat_theta, self.theta)
        self.assertTrue(rho > 0.95)
        self.assertTrue(pval < 1e-35)

    @unittest.skip("doesnt work right now")
    def test_joint_distribution_per_target_model(self):
        """
        Test the joint distributions match for a single column theta (not a per-target-nucleotide model)
        """
        theta = np.random.rand(self.feat_gen.feature_vec_len, NUM_NUCLEOTIDES) * 4
        theta_mask = get_possible_motifs_to_targets(self.feat_gen.get_motif_list(), theta.shape)
        theta[~theta_mask] = -np.inf

        rho, pval = self._test_joint_distribution(theta, theta)

        self.assertTrue(rho > 0.98)
        self.assertTrue(pval < 1e-37)
