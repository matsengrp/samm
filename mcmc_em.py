import numpy as np

from models import *
from feature_generator import SubmotifFeatureGenerator
from mutation_order_sampler import MutationOrderSampler
from survival_problem import SurvivalProblem

class MCMC_EM:
    def __init__(self, observed_data, submotif_len=3, num_e_samples=10):
        self.observed_data = observed_data
        self.feat_generator = SubmotifFeatureGenerator(submotif_len)
        self.num_e_samples = num_e_samples

    def run(self, max_iters=10):
        # initialize theta vector
        theta = np.random.randn(self.feat_generator.feature_vec_len)

        for steps in range(max_iters):
            # do E-step
            mut_order_sampler = MutationOrderSampler(
                theta,
                self.feat_generator,
            )
            samples = [
                ImputedSequenceMutations(
                    obs_seq,
                    mut_order_sampler.sample_order(obs_seq)
                )
                for obs_seq in self.observed_data
                for i in range(self.num_e_samples)
            ]

            # Do M-step
            problem = SurvivalProblem(samples)
            theta = problem.solve(self.feat_generator)
        return theta
