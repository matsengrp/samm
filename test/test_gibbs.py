import unittest
import itertools
import csv
import numpy as np
from scipy.stats import spearmanr
from collections import Counter

from common import NUM_NUCLEOTIDES
from models import ObservedSequenceMutations
from survival_model_simulator import SurvivalModelSimulator
from feature_generator import SubmotifFeatureGenerator
from mutation_order_gibbs import MutationOrderGibbsSampler, GibbsStepInfo

class MCMC_EM_TestCase(unittest.TestCase):
    def test_speedup(self):
        feat_generator = SubmotifFeatureGenerator(motif_len=3)
        theta = np.random.rand(feat_generator.feature_vec_len, NUM_NUCLEOTIDES)
        obs_seq_m = ObservedSequenceMutations("ttcgta", "tgggtt")
        gibbs_sampler = MutationOrderGibbsSampler(theta, feat_generator, obs_seq_m)

        curr_order = obs_seq_m.mutation_pos_dict.keys()
        curr_dicts, curr_seqs, curr_log_probs = gibbs_sampler._compute_log_probs(curr_order)

        new_order = [curr_order[0], curr_order[-1]] + curr_order[1:-1]
        feat_vec_dicts_slow, intermediate_seqs_slow, multinomial_sequence_slow = gibbs_sampler._compute_log_probs(new_order)
        feat_vec_dicts, intermediate_seqs, multinomial_sequence = gibbs_sampler._compute_log_probs(
            new_order,
            GibbsStepInfo(curr_order, curr_dicts, curr_seqs, curr_log_probs),
            update_positions=range(1, len(new_order)),
        )
        self.assertEqual(intermediate_seqs, intermediate_seqs_slow)
        self.assertEqual(feat_vec_dicts, feat_vec_dicts_slow)
        self.assertTrue(np.allclose(multinomial_sequence, multinomial_sequence_slow))

    def test_joint_distribution(self):
        """
        Check that the distribution of mutation orders is similar when we generate mutation orders directly
        from the survival model vs. when we generate mutation orders given the mutation positions from the
        gibbs sampler
        """
        np.random.seed(10)

        START_SEQ = "ttcg" # MUST BE LESS THAN TEN
        NUM_OBS_SAMPLES = 5000
        BURN_IN = 15
        CENSORING_TIME = 2.0
        LAMBDA0 = 0.1
        NUM_TOP_COMMON = 20

        feat_generator = SubmotifFeatureGenerator(motif_len=3)
        theta = np.random.rand(feat_generator.feature_vec_len, NUM_NUCLEOTIDES)
        surv_simulator = SurvivalModelSimulator(theta, feat_generator, lambda0=LAMBDA0)

        # Simulate some data from the same starting sequence
        # Get the distribution of mutation orders from our survival model
        full_seq_muts = [surv_simulator.simulate(START_SEQ, censoring_time=CENSORING_TIME) for i in range(NUM_OBS_SAMPLES)]
        # We make the mutation orders strings so easy to process
        true_order_distr = ["".join(map(str,m.get_mutation_order())) for m in full_seq_muts]
        obs_seq_mutations = [ObservedSequenceMutations(START_SEQ, m.obs_seq_mutation.end_seq) for m in full_seq_muts]

        # Now get the distribution of orders from our gibbs sampler (so sample mutation order
        # given known mutation positions)
        gibbs_order = []
        for i, obs_seq_m in enumerate(obs_seq_mutations):
            gibbs_sampler = MutationOrderGibbsSampler(np.max(theta, axis=1), feat_generator, obs_seq_m)
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

        self.assertTrue(rho > 0.94)
        self.assertTrue(pval < 1e-31)
