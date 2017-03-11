from collections import Counter
import numpy as np
import scipy.misc
from sampler_collection import SamplerCollection

class LogLikelihoodEvaluator:
    """
    Evaluates the likelihood of a set of model parameters for the given dataset
    """
    def __init__(self, obs_data, sampler_cls, feat_generator, num_jobs, scratch_dir):
        """
        @param obs_data: list of ObservedSequenceMutations
        @param sampler_cls: sampler class, actually has to be MutationOrderGibbsSampler
        @param feat_generator: SubmotifFeatureGenerator
        @param num_jobs: number of jobs to submit
        @param scratch_dir: tmp dir for batch submission manager
        """
        self.obs_data = obs_data
        self.init_orders = [obs_seq.mutation_pos_dict.keys() for obs_seq in obs_data]
        self.sampler_cls = sampler_cls
        self.feat_generator = feat_generator
        self.num_jobs = num_jobs
        self.scratch_dir = scratch_dir

    def _get_log_lik_obs_seq(self, sampler, sampled_orders):
        """
        Get the log likelihood of this ending sequence given the starting sequence
        @param sampler: MutationOrderGibbsSampler
        @param sampled_orders: the sampled orders from gibbs
        """
        assert(sampler.obs_seq_mutation.start_seq_with_flanks == sampled_orders.samples[0].obs_seq_mutation.start_seq_with_flanks)
        assert(sampler.obs_seq_mutation.end_seq_with_flanks == sampled_orders.samples[0].obs_seq_mutation.end_seq_with_flanks)
        obs_seq_samples = sampled_orders.samples

        # The sampled orders for this observed start/end sequence pair
        # log p(end|start,theta) = log p(reference order | start, theta) - log p(reference order | end, start, theta)

        # Use the middle order as a reference
        ref_sample_idx = len(obs_seq_samples)/2
        reference_order = obs_seq_samples[ref_sample_idx].mutation_order
        log_prob_ref_order = sampler.get_log_probs(reference_order)

        # Count number of times this order appears - this is our estimate of
        # p(reference order | end, start, theta)
        num_appears = np.sum([
            reference_order == sampled_order.mutation_order for sampled_order in obs_seq_samples
        ])
        print "num_appears", num_appears
        log_prob_order = np.log(float(num_appears)/len(obs_seq_samples))

        return log_prob_ref_order - log_prob_order

    def get_log_lik(self, theta, burn_in=0):
        """
        Get the log likelihood of the data
        @param theta: the model parameter to evaluate this for
        @param burn_in: number of burn in iterations for gibbs
        """
        sampler_collection = SamplerCollection(
            self.obs_data,
            theta,
            self.sampler_cls,
            self.feat_generator,
            num_jobs=self.num_jobs,
            scratch_dir=self.scratch_dir,
        )

        # Get samples drawn from the distribution P(order | start, end, theta)
        # If num_jobs > 1, will use srun to get jobs!
        # We need a lot of gibbs samples if the number of mutations is high. Let's calculate the number of mutations
        num_mutations_approx = int(np.mean([len(m) for m in self.init_orders[:10]]))
        sampler_results = sampler_collection.get_samples(
            self.init_orders,
            np.power(num_mutations_approx, 3), # Draw a lot of gibbs samples
            burn_in,
            get_full_sweep=True,
        )
        # Store the sampled orders for faster runs next time
        self.init_orders = [res.samples[-1].mutation_order for res in sampler_results]

        data_set_log_lik = [
            self._get_log_lik_obs_seq(sampler, sampled_orders)
            for sampler, sampled_orders in zip(sampler_collection.samplers, sampler_results)
        ]
        total_log_lik = np.sum(data_set_log_lik)
        return total_log_lik
