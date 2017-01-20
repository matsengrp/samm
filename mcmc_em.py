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

    # TODO: find a reasonable choice for upper_stop
    def run(self, max_iters=10, alpha=.05, upper_stop=1.):
        # initialize theta vector
        theta = np.random.randn(self.feat_generator.feature_vec_len)
        # stores the initialization for the gibbs samplers for the next iteration's e-step
        init_orders_for_iter = [obs_seq.mutation_pos_dict.keys() for obs_seq in self.observed_data]

        upper_bound_is_large = True
        run = 0
        all_lik_ratios = np.zeros(max_iters+self.burn_in)
        # TODO: use a separate burn-in value for computing ESS?
        while upper_bound_is_large and run < max_iters+self.burn_in:
            prev_samples = []
            lower_bound_is_negative = True
            while lower_bound_is_negative:
                prev_theta = theta
                num_e_samples = self.base_num_e_samples
                # do E-step, prepending previous samples
                samples_for_obs_seq = prev_samples + [
                    self.get_e_samples(obs_seq, theta, init_order, num_e_samples) for obs_seq, init_order in zip(self.observed_data, init_orders_for_iter)
                ]
                # use this iteration's sampled mutation orders as initialization for the gibbs samplers next cycle
                init_orders_for_iter = [samples[-1].mutation_order for samples in samples_for_obs_seq]
                e_step_samples = [s for s in samples for samples in samples_for_obs_seq]

                # Do M-step
                problem = SurvivalProblem(e_step_samples)
                theta, exp_log_lik = problem.solve(self.feat_generator)
    
                # Get statistics
                lik_ratio_mean, lik_ratio_var = problem.calculate_lik_stats(self.feat_generator, theta, prev_theta, e_step_samples)
                all_lik_ratios[run] = lik_ratio_mean

                if run > self.burn_in:
                    # Calculate lower bound to determine if we need to rerun
                    autocorr = self.calculate_autocorr(all_lik_ratios[:run])
                    neff = len(e_step_samples) / autocorr
                    ase = np.sqrt(lik_ratio_var / neff)
                    print run, autocorr, neff

                    # TODO: should we just have a table of z scores?
                    lower_bound = lik_ratio_mean - 1.65 * np.sqrt(lik_ratio_var / neff)
                    upper_bound = lik_ratio_mean + 1.65 * np.sqrt(lik_ratio_var / neff)
                    lower_bound_is_negative = (lower_bound < 0)
                    upper_bound_is_large = (upper_bound > upper_stop)
                    prev_samples = samples_for_obs_seq
                    
                    # Take enough samples next time to get large enough neff
                    assert(autocorr > 1)
                    num_e_samples = np.ceil((autocorr - 1) * num_e_samples)
                else:
                    # If still in burn-in go to next iteration
                    lower_bound_is_negative = False
            run += 1
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

    def calculate_autocorr(self, lik_ratios):
        # Definition from p. 151 of Carlin/Louis:
        # \kappa = 1 + 2\sum_{k=1}^\infty \rho_k
        # So we don't take the self-correlation
        # TODO: do we worry about cutting off small values?
        # Glynn/Whitt say we could use batch estimation with batch sizes going to
        # infinity. Is this a viable option?
        result = np.correlate(lik_ratios, lik_ratios, mode='full')
        return 1 + 2*np.sum(result[1+result.size/2:])

