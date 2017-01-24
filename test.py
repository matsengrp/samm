import numpy as np
import logging as log

from models import *
from common import *
from mcmc_em import MCMC_EM
from feature_generator import SubmotifFeatureGenerator
from simulate_from_survival import SurvivalModelSimulator
from mutation_order_gibbs import MutationOrderGibbsSampler

np.random.seed(10)

log.basicConfig(format="%(message)s", level=log.DEBUG)

num_threads = 12
seq_length = 50
num_extra_seq = 50
lasso_param = 0.1

feat_generator = SubmotifFeatureGenerator()
motif_list = SubmotifFeatureGenerator.get_motif_list()

# True vector
true_theta = np.matrix(np.zeros(feat_generator.feature_vec_len)).T
true_theta[0:4] = 50
true_theta[20:24] = 30
true_theta[43] = 70
true_theta[59] = 20
true_theta[63] = 20

# Simulate some data
simulator = SurvivalModelSimulator(true_theta, feat_generator, lambda0=0.1)
start_seqs = [
    "taagggaaaaaaaaaatgggggttt",
    "tttaaaaaagggggggggttgggttaaat",
    "gca" + motif_list[1] + "tagagta" + motif_list[2] + "atgatta",
    "gcat" + motif_list[1] + "tagt" + motif_list[3] + "tta",
    "gcagc" + motif_list[2] + "ttt" + motif_list[2] + "cca",
    "atgataa" + motif_list[0] + "ggatagg" + motif_list[3] + "tcctaggat",
    "atgacattccaa" + motif_list[21] + "gcgatccagcg" + motif_list[20] + "ttaggat",
    "atccaa" + motif_list[20] + "gcgcg" + motif_list[23] + "ttag",
    "atccaaggtt" + motif_list[22] + "gcgcaaccg" + motif_list[21] + "ttagtttag",
    "taa" + motif_list[22] + "ga" + motif_list[23] + "ttccat",
    "tttaggccaa" + motif_list[43] + "gccca" + motif_list[44] + "tagaccat",
    "ttaa" + motif_list[43] + "gca" + motif_list[22] + "tagat",
]
for i in range(num_extra_seq):
    start_seqs.append(
        get_random_dna_seq(seq_length)
    )

full_data_samples = [simulator.simulate(s, 0.7) for s in start_seqs]
obs_data = [s.obs_seq_mutation for s in full_data_samples]

# Run em
em_algo = MCMC_EM(
    obs_data,
    feat_generator,
    MutationOrderGibbsSampler,
    num_threads=num_threads
)
theta = em_algo.run(lasso_param=lasso_param)
log.info("==== FINAL theta ====")
for i in range(theta.size):
    log.info("%d: %f" % (i, theta[i]))
