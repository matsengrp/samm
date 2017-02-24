import time
from multiprocessing import Pool
import numpy as np
import scipy as sp

import logging as log
from survival_problem import SurvivalProblem
from submotif_feature_generator import SubmotifFeatureGenerator
from parallel_worker import MultiprocessingManager, ParallelWorker
from common import *
from profile_support import profile

class SamplePrecalcData:
    def __init__(self, features_per_step_matrix, init_grad_vector, mutating_pos_feat_vals):
        self.features_per_step_matrix = features_per_step_matrix
        self.init_grad_vector = init_grad_vector
        self.mutating_pos_feat_vals = mutating_pos_feat_vals

class SurvivalProblemCustom(SurvivalProblem):
    """
    Our own implementation to solve the survival problem
    """
    print_iter = 10 # print status every `print_iter` iterations

    def __init__(self, feat_generator, samples, penalty_param, theta_mask, num_threads=1):
        assert(penalty_param >= 0)
        self.feature_generator = feat_generator
        self.samples = samples
        self.theta_mask = theta_mask
        self.per_target_model = self.theta_mask.shape[1] == NUM_NUCLEOTIDES
        self.num_samples = len(self.samples)
        self.penalty_param = penalty_param
        self.pool = None
        self.num_threads = num_threads

        self.precalc_data = self._create_gradient_matrices(samples)

        self.post_init()

    def post_init(self):
        return

    def _create_gradient_matrices(self, samples):
        """
        Calculate the components in the gradient at the beginning of gradient descent
        Then the gradient can be calculated using element-wise matrix multiplication
        This is much faster than a for loop!

        We pre-calculate:
            1. `features_per_step_matrix`: the number of times each feature showed up at the mutation step
            2. `base_grad`: the gradient of the sum of the exp * psi terms
            3. `mutating_pos_feat_vals`: the feature idxs for which a mutation occured
        """
        precalc_data = []
        for sample in samples:
            feat_mut_steps = self.feature_generator.create_for_mutation_steps(sample)

            mutating_pos_feat_vals = []
            base_grad = np.zeros(self.feature_generator.feature_vec_len)
            # get the grad component from grad of psi * theta
            for feat_mut_step in feat_mut_steps:
                base_grad[feat_mut_step.mutating_pos_feat] += 1
                mutating_pos_feat_vals.append(feat_mut_step.mutating_pos_feat)

            # Get the grad component from grad of log(sum(exp(psi * theta)))
            # This matrix is just the number of times we saw each feature in the risk group

            features_per_step_matrix = np.zeros((
                self.feature_generator.feature_vec_len,
                sample.obs_seq_mutation.num_mutations
            ))

            features_per_step_matrix[:,0] = sample.obs_seq_mutation.feat_counts_flat
            prev_feat_mut_step = feat_mut_steps[0]
            for i, feat_mut_step in enumerate(feat_mut_steps[1:]):
                # All the features are very similar between risk groups - copy first
                features_per_step_matrix[:,i + 1] = features_per_step_matrix[:,i]

                # Remove feature corresponding to position that mutated already
                features_per_step_matrix[prev_feat_mut_step.mutating_pos_feat, i + 1] -= 1

                # Need to update the terms for positions near the previous mutation
                # Remove old feature values
                old_feat_idxs = feat_mut_step.neighbors_feat_old.values()
                features_per_step_matrix[old_feat_idxs, i + 1] -= 1
                # Add new feature values
                new_feat_idxs = feat_mut_step.neighbors_feat_new.values()
                features_per_step_matrix[new_feat_idxs, i + 1] += 1

                prev_feat_mut_step = feat_mut_step

            precalc_data.append(
                SamplePrecalcData(
                    features_per_step_matrix,
                    base_grad,
                    np.array(mutating_pos_feat_vals, dtype=int),
                )
            )
        return precalc_data

    def get_value(self, theta):
        """
        @return negative penalized log likelihood
        """
        raise NotImplementedError()

    def _get_log_lik_parallel(self, theta, batch_factor=1):
        """
        @param theta: the theta to calculate the likelihood for
        @param batch_factor: When using multiprocessing, we batch the samples together for speed
                            We make `num_threads` * `batch_factor` batches
        @return vector of log likelihood values
        """
        if self.pool is None:
            raise ValueError("Pool has not been initialized")

        rand_seed = get_randint()
        worker_list = [
            ObjectiveValueWorker(rand_seed + i, theta, sample_data)
            for i, sample_data in enumerate(self.precalc_data)
        ]
        if self.num_threads > 1:
            multiproc_manager = MultiprocessingManager(self.pool, worker_list, self.num_threads * batch_factor)
            ll = multiproc_manager.run()
        else:
            ll = [worker.run() for worker in worker_list]
        return np.array(ll)

    def _get_gradient_log_lik(self, theta, batch_factor=2):
        """
        @param theta: the theta to calculate the likelihood for
        @param batch_factor: When using multiprocessing, we batch the samples together for speed
                            We make `num_threads` * `batch_factor` batches

        Calculate the gradient of the negative log likelihood - delegates to separate cpu threads if threads > 1
        """
        if self.pool is None:
            raise ValueError("Pool has not been initialized")

        rand_seed = get_randint()
        worker_list = [
            GradientWorker(rand_seed + i, theta, sample_data)
            for i, sample_data in enumerate(self.precalc_data)
        ]
        if self.num_threads > 1:
            multiproc_manager = MultiprocessingManager(self.pool, worker_list, self.num_threads * batch_factor)
            l = multiproc_manager.run()
        else:
            l = [worker.run() for worker in worker_list]

        grad_ll_dtheta = np.sum(l, axis=0)
        return -1.0/self.num_samples * grad_ll_dtheta

    @staticmethod
    def calculate_per_sample_log_lik(theta, sample_data):
        """
        Calculate the log likelihood of this sample
        """
        exp_theta = np.exp(theta)
        risk_groups_exp_thetas = np.multiply(sample_data.features_per_step_matrix, exp_theta)
        denominators = risk_groups_exp_thetas.sum(axis=0)
        numerators = exp_theta[sample_data.mutating_pos_feat_vals]
        log_lik = np.log(numerators).sum() - np.log(denominators).sum()
        return log_lik

    @staticmethod
    def get_gradient_log_lik_per_sample(theta, sample_data):
        """
        Calculate the gradient of the log likelihood of this sample
        All the gradients for each step are the gradient of psi * theta - log(sum(exp(theta * psi)))
        Calculate the gradient from each step one at a time

        @param theta: the theta to evaluate the gradient at
        @param sample_data: SamplePrecalcData
        """
        grad_log_sum_exps = np.multiply(sample_data.features_per_step_matrix, np.exp(theta))
        denominators = grad_log_sum_exps.sum(axis=0)
        grad_components = np.divide(grad_log_sum_exps, denominators)
        grad = sample_data.init_grad_vector - grad_components.sum(axis=1)
        return np.reshape(grad, (grad.size, 1))

class GradientWorker(ParallelWorker):
    """
    Stores the information for calculating gradient
    """
    def __init__(self, seed, theta, sample_data):
        """
        @param theta: where to take the gradient of the total log likelihood
        @param sample: class ImputedSequenceMutations
        @param feature_vecs: list of sparse feature vectors for all at-risk positions at every mutation step
        """
        self.seed = seed
        self.theta = theta
        self.sample_data = sample_data

    def run_worker(self):
        """
        @return the gradient of the log likelihood for this sample
        """
        return SurvivalProblemCustom.get_gradient_log_lik_per_sample(self.theta, self.sample_data)

class ObjectiveValueWorker(ParallelWorker):
    """
    Stores the information for calculating objective function value
    """
    def __init__(self, seed, theta, sample_data):
        """
        @param theta: where to take the gradient of the total log likelihood
        @param sample: SamplePrecalcData
        """
        self.seed = seed
        self.theta = theta
        self.sample_data = sample_data

    def run_worker(self):
        """
        @return the log likelihood for this sample
        """
        return SurvivalProblemCustom.calculate_per_sample_log_lik(self.theta, self.sample_data)
