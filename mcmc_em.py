import time
import numpy as np

from models import *
from feature_generator import SubmotifFeatureGenerator
# from mutation_order_sampler import MutationOrderSampler
from mutation_order_gibbs import MutationOrderGibbsSampler
from survival_problem import SurvivalProblem
from sampler_collection import SamplerCollection

class MCMC_EM:
    def __init__(self, observed_data, feat_generator, sampler_cls, base_num_e_samples=10, burn_in=10, num_threads=1):
        self.observed_data = observed_data
        self.feat_generator = feat_generator
        self.base_num_e_samples = base_num_e_samples
        self.burn_in = burn_in
        self.sampler_cls = sampler_cls
        self.num_threads = num_threads

    def run(self, max_iters=10, verbose=False):
        # initialize theta vector
        theta = np.random.randn(self.feat_generator.feature_vec_len)
        # stores the initialization for the gibbs samplers for the next iteration's e-step
        init_orders = [obs_seq.mutation_pos_dict.keys() for obs_seq in self.observed_data]
        for i in range(max_iters):
            # do E-step
            num_e_samples = (i + 1) * self.base_num_e_samples
            sampler_collection = SamplerCollection(
                self.observed_data,
                theta,
                self.sampler_cls,
                self.feat_generator,
                self.num_threads,
            )

            sampled_orders_list = sampler_collection.get_samples(
                init_orders,
                num_e_samples,
                self.burn_in,
            )
            # the last sampled mutation order from each list
            # use this iteration's sampled mutation orders as initialization for the gibbs samplers next cycle
            init_orders = [sampled_orders[-1].mutation_order for sampled_orders in sampled_orders_list]
            # flatten the list of samples to get all the samples
            e_step_samples = [o for orders in sampled_orders_list for o in orders]

            # Do M-step
            problem = SurvivalProblem(e_step_samples)
            theta, exp_log_lik = problem.solve(self.feat_generator, verbose=verbose)
        return theta

    def get_e_samples(self, obs_seq, theta, init_order, num_samples):
        mut_order_sampler = MutationOrderGibbsSampler(
            theta,
            self.feat_generator,
            obs_seq
        )
        # TODO: we can also try different amount of burn in for each EM iteration in the future
        mut_order_sampler.sample_burn_in(init_order, self.burn_in)
        # TODO: we definitely want different number of samples at each E-step in the future
        sampled_orders = mut_order_sampler.sample_orders(num_samples)
        return [ImputedSequenceMutations(obs_seq, order) for order in sampled_orders]
