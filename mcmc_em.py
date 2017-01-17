import numpy as np

from models import *
from feature_generator import SubmotifFeatureGenerator
# from mutation_order_sampler import MutationOrderSampler
from mutation_order_gibbs import MutationOrderGibbsSampler
from survival_problem import SurvivalProblem

class MCMC_EM:
    def __init__(self, observed_data, feat_generator, base_num_e_samples=10, burn_in=10):
        self.observed_data = observed_data
        self.feat_generator = feat_generator
        self.base_num_e_samples = base_num_e_samples
        self.burn_in = burn_in

    def run(self, max_iters=10):
        # initialize theta vector
        theta = np.random.randn(self.feat_generator.feature_vec_len)
        # stores the initialization for the gibbs samplers for the next iteration's e-step
        init_orders_for_iter = [obs_seq.mutation_pos_dict.keys() for obs_seq in self.observed_data]
        for i in range(max_iters):
            num_e_samples = (i + 1) * self.base_num_e_samples
            # do E-step
            samples_for_obs_seq = [
                self.get_e_samples(obs_seq, theta, init_order, num_e_samples) for obs_seq, init_order in zip(self.observed_data, init_orders_for_iter)
            ]
            # use this iteration's sampled mutation orders as initialization for the gibbs samplers next cycle
            init_orders_for_iter = [samples[-1].mutation_order for samples in samples_for_obs_seq]
            e_step_samples = [s for s in samples for samples in samples_for_obs_seq]
            # Do M-step
            problem = SurvivalProblem(e_step_samples)
            theta, exp_log_lik = problem.solve(self.feat_generator)
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
