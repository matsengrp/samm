import unittest
import itertools
import csv
import numpy as np
from scipy.stats import spearmanr
from collections import Counter

from models import ObservedSequenceMutations
from survival_model_simulator import SurvivalModelSimulator
from feature_generator import SubmotifFeatureGenerator
from mutation_order_gibbs import MutationOrderGibbsSampler

class MCMC_EM_TestCase(unittest.TestCase):
    def test_joint_distribution(self):
        np.random.seed(10)

        START_SEQ = "ttcgat" # MUST BE LESS THAN TEN
        NUM_OBS_SAMPLES = 3000
        BURN_IN = 15
        CENSORING_TIME = 4.0
        LAMBDA0 = 0.1
        NUM_TOP_COMMON = 20

        feat_generator = SubmotifFeatureGenerator(submotif_len=3)
        theta = np.random.rand(feat_generator.feature_vec_len)
        surv_simulator = SurvivalModelSimulator(theta, feat_generator, lambda0=LAMBDA0)
        full_seq_muts = [surv_simulator.simulate(START_SEQ, censoring_time=CENSORING_TIME) for i in range(NUM_OBS_SAMPLES)]
        true_order_distr = ["".join(map(str,m.get_mutation_order())) for m in full_seq_muts]
        obs_seq_mutations = [ObservedSequenceMutations(START_SEQ, m.obs_seq_mutation.end_seq) for m in full_seq_muts]

        gibbs_order = []
        for i, obs_seq_m in enumerate(obs_seq_mutations):
            gibbs_sampler = MutationOrderGibbsSampler(theta, feat_generator, obs_seq_m)
            gibbs_samples = gibbs_sampler.run(obs_seq_m.mutation_pos_dict.keys(), BURN_IN, 1)
            order_sample = gibbs_samples[0].mutation_order
            order_sample = map(str, order_sample)
            gibbs_order.append("".join(order_sample))

        true_counter = Counter(true_order_distr)
        gibbs_counter = Counter(gibbs_order)
        all_orders = set(true_counter.keys()).union(set(gibbs_counter.keys()))
        true_counter_order_vec = [max(true_counter.get(o), 0) for o in all_orders]
        gibbs_counter_order_vec = [max(gibbs_counter.get(o), 0) for o in all_orders]

        rho, pval = spearmanr(true_counter_order_vec, gibbs_counter_order_vec)
        print "rho, pval", rho, pval

        for t, g in zip(true_counter.most_common(NUM_TOP_COMMON), gibbs_counter.most_common(NUM_TOP_COMMON)):
            print "%s (%d) \t %s (%d)" % (t[0], t[1], g[0], g[1])

        self.assertTrue(rho > 0.15)
        self.assertTrue(pval < 1e-5)
