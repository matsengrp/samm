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

        # p_reforder is the estimate for p(end|start,theta) using a particular reference order. We can calculate it as follows:
        # log p_reforder(end|start,theta) = log p(reference order | start, theta) - log p(reference order | end, start, theta)
        # The first log prob term is computed analytically (it's not conditional on the end sequence, so easy to calculate)
        # The second log prob term is estimated using the empirical distribution of orders from the gibbs sampler
        # We estimate log p(end|start,theta) by taking an average over all orders observed from the gibbs sampler
        # So log p(end|start,theta) = log(mean(p_reforder(end|start,theta)))

        count_dict = {}
        for s in obs_seq_samples:
            mut_order_str = ".".join(map(str, s.mutation_order))
            if mut_order_str not in count_dict:
                count_dict[mut_order_str] = 1
            else:
                count_dict[mut_order_str] += 1

        log_probs = []
        num_sampled_orders = len(count_dict)
        num_samples = len(obs_seq_samples)
        for order_str, order_cnt in count_dict.iteritems():
            reference_order = [int(p) for p in order_str.split(".")]
            log_prob_ref_order = sampler.get_log_probs(reference_order)

            # Count number of times this order appears - this is our estimate of
            # p(reference order | end, start, theta)
            log_prob_order = np.log(float(order_cnt)/num_samples)
            log_probs.append(log_prob_ref_order - log_prob_order)

        log_mean_prob = np.log(np.exp(scipy.misc.logsumexp(log_probs))/num_sampled_orders)
        return log_mean_prob

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
            num_mutations_approx, # Draw a lot of gibbs samples
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
