import time
import numpy as np
from sampler_collection import SamplerCollection
from mutation_order_gibbs import MutationOrderGibbsSampler
from mutation_order_chibs import MutationOrderChibsSampler
from survival_problem_lasso import SurvivalProblemLasso
import logging as log
from common import *

class GreedyLikelihoodComparer:
    """
    Given a list of models to compare, this will use a greedy method to compare them
    """
    @staticmethod
    def do_greedy_search(val_set, feat_generator, models, sort_func, burn_in, num_samples, num_jobs, scratch_dir, pool=None):
        """
        @param val_set: list of ObservedSequenceMutations
        @param feat_generator: FeatureGenerator
        @param models: list of MethodResults
        @param sort_func: a function for sorting the list of models, some measure of model complexity
        @param burn_in, num_samples, num_jobs, scratch_dir

        @return Orders the models from least complex to the most complex. Stops searching the model list once the difference
            between EM surrogate functions is negative. Returns the last model where the EM surrogate function was increasing.
        """
        sorted_models = sorted(models, key=sort_func)
        best_model = sorted_models[0]
        best_model_idx = 0
        val_set_evaluator = LikelihoodComparer(
            val_set,
            feat_generator,
            theta_ref=best_model.penalized_theta,
            num_samples=num_samples,
            burn_in=burn_in,
            num_jobs=num_jobs,
            scratch_dir=scratch_dir,
            pool=pool,
        )
        for model_idx, model in enumerate(sorted_models[1:]):
            log_lik_ratio, lower_bound, upper_bound = val_set_evaluator.get_log_likelihood_ratio(model.penalized_theta)
            log.info("  Greedy search: ratio %f (%f, %f), model %s" % (log_lik_ratio, lower_bound, upper_bound, model))
            if log_lik_ratio >= 0:
                best_model = model
                best_model_idx = model_idx + 1
                if model_idx != len(sorted_models) - 2:
                    val_set_evaluator = LikelihoodComparer(
                        val_set,
                        feat_generator,
                        theta_ref=best_model.penalized_theta,
                        num_samples=num_samples,
                        burn_in=burn_in,
                        num_jobs=num_jobs,
                        scratch_dir=scratch_dir,
                        pool=pool,
                    )
            else:
                break
        return best_model, best_model_idx

class LikelihoodComparer:
    """
    Compares the likelihood of between model parameters for the given dataset

    Determines whether theta vs. theta_ref have a higher marginal likelihood

    Recall from EM that
    log marginal likelihood(theta) - log marginal likelihood(theta ref) >= Q(theta | theta ref) - Q(theta ref | theta ref)
    where Q = E[log lik(theta | data)]

    Therefore we can compare theta parameters using the Q function
    """
    MAX_TOT_SAMPLES = 50000

    def __init__(self, obs_data, feat_generator, theta_ref, num_samples=10, burn_in=0, num_jobs=1, scratch_dir="", pool=None):
        """
        @param obs_data: list of ObservedSequenceMutations
        @param feat_generator: SubmotifFeatureGenerator
        @param theta_ref: the model parameters (numpy vector) - we use this as the reference theta
        @param num_samples: number of samples to draw for the likelihood ratio estimation
        @param burn_in: number of burn in samples
        @param num_jobs: number of jobs to submit
        @param scratch_dir: tmp dir for batch submission manager
        @param pool: multiprocessing pool previously initialized before model fitting
        """
        self.theta_ref = theta_ref
        self.num_samples = num_samples
        self.feat_generator = feat_generator
        self.per_target_model = theta_ref.shape[1] == NUM_NUCLEOTIDES + 1
        self.pool = pool

        log.info("Creating likelihood comparer")
        st_time = time.time()
        self.sampler_collection = SamplerCollection(
            obs_data,
            self.theta_ref,
            MutationOrderGibbsSampler,
            feat_generator,
            num_jobs=num_jobs,
            scratch_dir=scratch_dir,
        )

        # Get samples drawn from the distribution P(order | start, end, theta reference)
        self.num_tot_obs = len(obs_data)
        self.init_orders = [
            np.random.permutation(obs_seq.mutation_pos_dict.keys()).tolist()
            for obs_seq in obs_data
        ]
        sampler_results = self.sampler_collection.get_samples(
            self.init_orders,
            self.num_samples,
            burn_in,
        )
        log.info("Finished getting samples, time %s" % (time.time() - st_time))
        sampled_orders_list = [res.samples for res in sampler_results]
        self.init_orders = [sampled_orders[-1].mutation_order for sampled_orders in sampled_orders_list]
        self.samples = [o for orders in sampled_orders_list for o in orders]
        self.sample_labels = [i for i, orders in enumerate(sampled_orders_list) for o in orders]
        # Setup a problem so that we can extract the log likelihood ratio
        st_time = time.time()
        self.prob = SurvivalProblemLasso(
            feat_generator,
            self.samples,
            sample_labels=self.sample_labels,
            penalty_params=[0],
            per_target_model=self.per_target_model,
            pool=self.pool,
        )
        log.info("Finished calculating sample info, time %s" % (time.time() - st_time))

    def get_log_likelihood_ratio(self, theta, max_iters=3):
        """
        Get the log likelihood ratio between theta and a reference theta
        @param theta: the model parameter to compare against
        @return Q(theta | theta ref) - Q(theta ref | theta ref)
        """
        ll_ratio_vec = self.prob.calculate_log_lik_ratio_vec(theta, self.theta_ref, group_by_sample=True)
        mean_ll_ratio = np.mean(ll_ratio_vec)
        ase, lower_bound, upper_bound, ess = get_standard_error_ci_corrected(ll_ratio_vec, ZSCORE_95, mean_ll_ratio)
        log.info("Likelihood comparer (lower,mean,upper, ess)=(%f,%f,%f,%f)" % (lower_bound, mean_ll_ratio, upper_bound, ess))

        curr_iter = 1
        while lower_bound < 0 and upper_bound > 0 and self.num_samples * (1 + curr_iter) * self.num_tot_obs < LikelihoodComparer.MAX_TOT_SAMPLES and curr_iter < max_iters:
            # If we aren't sure if the mean log likelihood ratio is negative or positive, grab more samples
            log.info("Get more samples likelihood comparer (lower,mean,upper,ess)=(%f,%f,%f,%f)" % (lower_bound, mean_ll_ratio, upper_bound, ess))
            st_time = time.time()
            sampler_results = self.sampler_collection.get_samples(
                self.init_orders,
                self.num_samples,
                burn_in_sweeps=0,
            )
            log.info("Finished getting samples, time %s" % (time.time() - st_time))
            sampled_orders_list = [res.samples for res in sampler_results]
            self.init_orders = [sampled_orders[-1].mutation_order for sampled_orders in sampled_orders_list]
            self.samples += [s for res in sampler_results for s in res.samples]
            self.sample_labels += [i for i, orders in enumerate(sampled_orders_list) for o in orders]
            self.num_samples += self.num_samples
            # Setup a problem so that we can extract the log likelihood ratio
            self.prob = SurvivalProblemLasso(
                self.feat_generator,
                self.samples,
                sample_labels=self.sample_labels,
                penalty_params=[0],
                per_target_model=self.per_target_model,
                pool=self.pool,
            )
            ll_ratio_vec = self.prob.calculate_log_lik_ratio_vec(theta, self.theta_ref, group_by_sample=True)
            mean_ll_ratio = np.mean(ll_ratio_vec)
            ase, lower_bound, upper_bound, ess = get_standard_error_ci_corrected(ll_ratio_vec, ZSCORE_95, mean_ll_ratio)
            curr_iter += 1

        return mean_ll_ratio, lower_bound, upper_bound

class LogLikelihoodEvaluator:
    """
    Evaluates the likelihood of a set of model parameters for the given dataset
    Uses Chibs
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

    def get_log_lik(self, theta, num_samples=1000, burn_in=0, num_tries=5):
        """
        Get the log likelihood of the data
        @param theta: the model parameter to evaluate this for
        @param num_samples: number of gibbs samples
        @param burn_in: number of burn in iterations for gibbs
        @param num_tries: number of tries for Chibs sampler
        """
        sampler_collection = SamplerCollection(
            self.obs_data,
            theta,
            MutationOrderChibsSampler,
            self.feat_generator,
            num_jobs=self.num_jobs,
            scratch_dir=self.scratch_dir,
            num_tries=num_tries,
        )

        # Get samples drawn from the distribution P(order | start, end, theta)
        # If num_jobs > 1, will use srun to get jobs!
        # We need a lot of gibbs samples if the number of mutations is high. Let's calculate the number of mutations
        sampler_results = sampler_collection.get_samples(
            self.init_orders,
            num_samples,
            burn_in,
            sampling_rate=0,
        )
        # Store the sampled orders for faster runs next time
        self.init_orders = [res.gibbs_samples[-1].mutation_order for res in sampler_results]

        data_set_log_lik = [res.log_prob_order - res.log_prob_estimate for res in sampler_results]
        total_log_lik = np.sum(data_set_log_lik)
        return total_log_lik
