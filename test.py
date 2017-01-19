import numpy as np

from models import *
from common import *
from mcmc_em import MCMC_EM
from feature_generator import SubmotifFeatureGenerator
from simulate_from_survival import SurvivalModelSimulator
from mutation_order_gibbs import MutationOrderGibbsSampler

np.random.seed(10)

num_threads = 2
seq_length = 15
num_extra_seq = 0

feat_generator = SubmotifFeatureGenerator()

# True vector
true_theta = np.matrix(np.zeros(feat_generator.feature_vec_len)).T
true_theta[0] = 100
true_theta[63] = 100

# Simulate some data
simulator = SurvivalModelSimulator(true_theta, 0.1, feat_generator)
start_seqs = [
    "taagggaaaaaaaaaatgggggttt",
    "tttaaaaaagggggggggttgggttaaat",
]
for i in range(num_extra_seq):
    start_seqs.append(
        get_random_dna_seq(seq_length)
    )

full_data_samples = [simulator.simulate(s, 0.1) for s in start_seqs]
obs_data = [s.obs_seq_mutation for s in full_data_samples]
for i, obs in enumerate(obs_data):
    for k, v in obs.mutation_pos_dict.iteritems():
        print k, obs.start_seq[k], "=>", v

# Run em
em_algo = MCMC_EM(
    obs_data,
    feat_generator,
    MutationOrderGibbsSampler,
)
theta = em_algo.run()
print "theta", theta
