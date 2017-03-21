import numpy as np
from sampler_collection import SamplerCollection
from mutation_order_chibs import MutationOrderChibsSampler

class LogLikelihoodEvaluator:
    """
    Evaluates the likelihood of a set of model parameters for the given dataset
    """

    def __init__(self, obs_data, feat_generator, num_jobs=1, scratch_dir=""):
        """
        @param obs_data: list of ObservedSequenceMutations
        @param feat_generator: SubmotifFeatureGenerator
        @param num_jobs: number of jobs to submit
        @param scratch_dir: tmp dir for batch submission manager
        """
        self.obs_data = obs_data
        self.init_orders = [obs_seq.mutation_pos_dict.keys() for obs_seq in obs_data]
        self.feat_generator = feat_generator
        self.num_jobs = num_jobs
        self.scratch_dir = scratch_dir

    def get_log_lik(self, theta, num_samples=1000, burn_in=0):
        """
        Get the log likelihood of the data
        @param theta: the model parameter to evaluate this for
        @param burn_in: number of burn in iterations for gibbs
        """
        sampler_collection = SamplerCollection(
            self.obs_data,
            theta,
            MutationOrderChibsSampler,
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
            num_samples,
            burn_in,
            get_full_sweep=True,
        )
        # Store the sampled orders for faster runs next time
        self.init_orders = [res.gibbs_samples[-1].mutation_order for res in sampler_results]

        data_set_log_lik = [res.log_prob_order - res.log_prob_estimate for res in sampler_results]
        total_log_lik = np.sum(data_set_log_lik)
        return total_log_lik
