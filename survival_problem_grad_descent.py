import time
from multiprocessing import Pool
import numpy as np
import scipy as sp
import logging as log
from survival_problem import SurvivalProblem
from submotif_feature_generator import SubmotifFeatureGenerator
from parallel_worker import *
from common import *
from profile_support import profile

class SurvivalProblemCustom(SurvivalProblem):
    """
    Our own implementation to solve the survival problem
    """
    print_iter = 10 # print status every `print_iter` iterations

    # TODO: Think about grouping samples together if they are from the same gibbs sampler
    def __init__(self, feat_generator, samples, penalty_param, theta_mask, num_threads=1):
        assert(penalty_param >= 0)
        self.feature_generator = feat_generator
        self.motif_len = feat_generator.motif_len
        self.samples = samples
        self.theta_mask = theta_mask
        self.per_target_model = self.theta_mask.shape[1] == NUM_NUCLEOTIDES
        self.num_samples = len(self.samples)
        self.feature_mut_steps_pair = [
            (sample, self.feature_generator.create_for_mutation_steps(sample))
            for sample in samples
        ]
        self.penalty_param = penalty_param
        self.pool = None
        self.num_threads = num_threads

        self.post_init()

    def post_init(self):
        return

    def get_value(self, theta):
        """
        @return negative penalized log likelihood
        """
        raise NotImplementedError()

    def get_log_lik(self, theta):
        """
        @return negative penalized log likelihood
        """
        log_lik = np.sum([
            SurvivalProblemCustom.calculate_per_sample_log_lik(theta, sample, feature_mut_steps)
            for sample, feature_mut_steps in self.feature_mut_steps_pair
        ])
        return 1.0/self.num_samples * log_lik

    def _get_log_lik_parallel(self, theta):
        """
        @return vector of log likelihood values
        """
        if self.pool is None:
            raise ValueError("Pool has not been initialized")

        rand_seed = get_randint()
        worker_list = [
            ObjectiveValueWorker(rand_seed + i, theta, sample, feature_vecs)
            for i, (sample, feature_vecs) in enumerate(self.feature_mut_steps_pair)
        ]
        if self.num_threads > 1:
            ll = self.pool.map(run_multiprocessing_worker, worker_list)
        else:
            ll = map(run_multiprocessing_worker, worker_list)
        return ll

    def _get_gradient_log_lik(self, theta):
        """
        Calculate the gradient - delegates to separate cpu threads if threads > 1
        """
        if self.pool is None:
            raise ValueError("Pool has not been initialized")

        rand_seed = get_randint()
        worker_list = [
            GradientWorker(rand_seed + i, theta, sample, feature_vecs)
            for i, (sample, feature_vecs) in enumerate(self.feature_mut_steps_pair)
        ]
        if self.num_threads > 1:
            l = self.pool.map(run_multiprocessing_worker, worker_list)
        else:
            l = map(run_multiprocessing_worker, worker_list)

        grad_ll_dtheta = np.sum(l, axis=0)
        return -1.0/self.num_samples * grad_ll_dtheta

    @staticmethod
    def calculate_per_sample_log_lik(theta, sample, feature_mutation_steps):
        """
        Calculate the log likelihood of this sample
        """
        # Get the components -- numerators and the denominators
        log_numerators = [np.asscalar(theta[mut_step.mutating_pos_feat]) for mut_step in feature_mutation_steps]
        denominators = [
            (np.exp(sample.obs_seq_mutation.feat_matrix_start * theta)).sum()
        ]
        for i, feat_mut_step in enumerate(feature_mutation_steps[1:]):
            old_denominator = denominators[i]
            old_log_numerator = log_numerators[i]
            old_feat_theta_sums = theta[feat_mut_step.neighbors_feat_old.values()]
            new_feat_theta_sums = theta[feat_mut_step.neighbors_feat_new.values()]
            new_denom = old_denominator - np.exp(old_log_numerator) - (np.exp(old_feat_theta_sums)).sum() + (np.exp(new_feat_theta_sums)).sum()
            denominators.append(new_denom)
        log_lik = sum(log_numerators) - np.sum(np.log(denominators))
        return log_lik

    @staticmethod
    def get_gradient_log_lik_per_sample(theta, sample, feature_mutation_steps):
        """
        Calculate the gradient of the log likelihood of this sample
        All the gradients for each step are the gradient of psi * theta - log(sum(exp(theta * psi)))
        Calculate the gradient from each step one at a time

        @param theta: the theta to evaluate the gradient at
        @param sample: ImputedSequenceMutations
        @param feature_mutation_steps: a list of FeatureMutationStep
        """
        exp_theta = np.exp(theta)

        # Calculate the gradient associated with the first mutation step
        exp_terms = exp_theta[sample.obs_seq_mutation.feat_dict_vals_start,]

        old_denom = exp_terms.sum()
        grad_log_sum_exp = sample.obs_seq_mutation.feat_matrix_startT * exp_terms

        grad = -1.0/old_denom * grad_log_sum_exp
        grad[feature_mutation_steps[0].mutating_pos_feat] += 1

        # Calculate the gradients associated with the rest of the mutation steps
        prev_feat_mut_step = feature_mutation_steps[0]
        for i, feat_mut_step in enumerate(feature_mutation_steps[1:]):
            old_numerator = exp_theta[prev_feat_mut_step.mutating_pos_feat]
            # Need to remove the previously mutated position from the risk group,
            # Also need to remove it from the gradient
            grad_log_sum_exp[prev_feat_mut_step.mutating_pos_feat] -= old_numerator

            # Need to update the terms for positions near the previous mutation
            old_feat_idxs = feat_mut_step.neighbors_feat_old.values()
            old_exp_thetas = exp_theta[old_feat_idxs]
            grad_log_sum_exp[old_feat_idxs] -= old_exp_thetas

            new_feat_idxs = feat_mut_step.neighbors_feat_new.values()
            new_exp_thetas = exp_theta[new_feat_idxs]
            grad_log_sum_exp[new_feat_idxs] += new_exp_thetas

            # Now update the denominator
            new_denom = old_denom - old_numerator - old_exp_thetas.sum() + new_exp_thetas.sum()

            # Finally update the gradient of psi * theta - log(sum(exp(theta * psi)))
            grad -= grad_log_sum_exp/new_denom
            # TODO: this doesn't work for theta more than one column
            grad[feat_mut_step.mutating_pos_feat] += 1

            old_denom = new_denom
            prev_feat_mut_step = feat_mut_step
        return grad

class GradientWorker(ParallelWorker):
    """
    Stores the information for calculating gradient
    """
    def __init__(self, seed, theta, sample, feature_vecs):
        """
        @param theta: where to take the gradient of the total log likelihood
        @param sample: class ImputedSequenceMutations
        @param feature_vecs: list of sparse feature vectors for all at-risk positions at every mutation step
        """
        self.seed = seed
        self.theta = theta
        self.sample = sample
        self.feature_vecs = feature_vecs

    def run(self):
        """
        @return the gradient of the log likelihood for this sample
        """
        return SurvivalProblemCustom.get_gradient_log_lik_per_sample(self.theta, self.sample, self.feature_vecs)

class ObjectiveValueWorker(ParallelWorker):
    """
    Stores the information for calculating objective function value
    """
    def __init__(self, seed, theta, sample, feature_vecs):
        """
        @param theta: where to take the gradient of the total log likelihood
        @param sample: class ImputedSequenceMutations
        @param feature_vecs: list of sparse feature vectors for all at-risk positions at every mutation step
        """
        self.seed = seed
        self.theta = theta
        self.sample = sample
        self.feature_vecs = feature_vecs

    def run(self):
        """
        @return the log likelihood for this sample
        """
        return SurvivalProblemCustom.calculate_per_sample_log_lik(self.theta, self.sample, self.feature_vecs)
