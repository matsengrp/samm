import numpy as np
from sampler_collection import SamplerCollection
from mutation_order_gibbs import MutationOrderGibbsSampler
from survival_problem_lasso import SurvivalProblemLasso
from common import *

class LikelihoodComparer:
    """
    Compares the likelihood of between model parameters for the given dataset

    Determines whether theta vs. theta_ref have a higher marginal likelihood

    Recall from EM that
    log marginal likelihood(theta) - log marginal likelihood(theta ref) >= Q(theta | theta ref) - Q(theta ref | theta ref)
    where Q = E[log lik(theta | data)]

    Therefore we can compare theta parameters using the Q function
    """

    def __init__(self, obs_data, feat_generator, theta_ref, num_samples=100, burn_in=0, num_jobs=1, scratch_dir=""):
        """
        @param obs_data: list of ObservedSequenceMutations
        @param feat_generator: SubmotifFeatureGenerator
        @param theta_ref: the model parameters (numpy vector) - we use this as the reference theta
        @param num_samples: number of samples to draw for the likelihood ratio estimation
        @param burn_in: number of burn in samples
        @param num_jobs: number of jobs to submit
        @param scratch_dir: tmp dir for batch submission manager
        """
        self.theta_ref = theta_ref
        per_target_model = theta_ref.shape[1] == NUM_NUCLEOTIDES

        sampler_collection = SamplerCollection(
            obs_data,
            self.theta_ref,
            MutationOrderGibbsSampler,
            feat_generator,
            num_jobs=num_jobs,
            scratch_dir=scratch_dir,
        )

        # Get samples drawn from the distribution P(order | start, end, theta reference)
        init_orders = [obs_seq.mutation_pos_dict.keys() for obs_seq in obs_data]
        sampler_results = sampler_collection.get_samples(
            init_orders,
            num_samples,
            burn_in,
        )

        # Setup a problem so that we can extract the log likelihood ratio
        self.prob = SurvivalProblemLasso(
            feat_generator,
            [o for res in sampler_results for o in res.samples],
            penalty_params=[0],
            per_target_model=per_target_model,
            theta_mask=None,
            num_threads=1,
        )

    def get_log_likelihood_ratio(self, theta):
        """
        Get the log likelihood ratio between theta and a reference theta
        @param theta: the model parameter to compare against
        @return Q(theta | theta ref) - Q(theta ref | theta ref)
        """
        ll_ratio_vec = self.prob.calculate_log_lik_ratio_vec(theta, self.theta_ref)
        return np.sum(ll_ratio_vec)
