import time
from multiprocessing import Pool
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix

import logging as log
from survival_problem import SurvivalProblem
from submotif_feature_generator import SubmotifFeatureGenerator
from parallel_worker import MultiprocessingManager, ParallelWorker
from common import *
from profile_support import profile

class SamplePrecalcData:
    def __init__(self, init_feat_counts, features_per_step_matrixT, init_grad_vector, mutating_pos_feat_vals):
        self.init_feat_counts = init_feat_counts
        self.features_per_step_matrixT = features_per_step_matrixT
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

        self.num_threads = num_threads
        self.pool = Pool(self.num_threads)

        self.precalc_data = self._create_precalc_data_parallel(samples)

        self.post_init()

    def post_init(self):
        return

    def _create_precalc_data_parallel(self, samples, batch_factor=10):
        """
        calculate the precalculated data for each sample in parallel
        """
        if self.pool is None:
            raise ValueError("Pool has not been initialized")

        rand_seed = get_randint()
        worker_list = [
            PrecalcDataWorker(
                rand_seed + i,
                sample,
                self.feature_generator.create_for_mutation_steps(sample),
                self.feature_generator.feature_vec_len,
            ) for i, sample in enumerate(samples)
        ]
        if self.num_threads > 1:
            multiproc_manager = MultiprocessingManager(self.pool, worker_list, self.num_threads * batch_factor)
            precalc_data = multiproc_manager.run()
        else:
            precalc_data = [worker.run() for worker in worker_list]
        return precalc_data

    def get_value(self, theta):
        """
        @return negative penalized log likelihood
        """
        raise NotImplementedError()

    def calculate_log_lik_ratio_vec(self, theta, prev_theta):
        llr_vec = np.zeros(self.num_samples)
        for sample_id, sample_data in enumerate(self.precalc_data):
            llr_vec[sample_id] = SurvivalProblemCustom.calculate_per_sample_log_lik(theta, sample_data) - \
                    SurvivalProblemCustom.calculate_per_sample_log_lik(prev_theta, sample_data)
        return llr_vec

    def _get_log_lik_parallel(self, theta, batch_factor=1):
        """
        @param theta: the theta to calculate the likelihood for
        @param batch_factor: When using multiprocessing, we batch the samples together for speed
                            We make `num_threads` * `batch_factor` batches
        @return negative log likelihood
        """
        if self.pool is None:
            raise ValueError("Pool has not been initialized")

        exp_theta = np.exp(theta)
        rand_seed = get_randint()
        worker_list = [
            ObjectiveValueWorker(rand_seed + i, exp_theta, sample_data)
            for i, sample_data in enumerate(self.precalc_data)
        ]
        if self.num_threads > 1:
            multiproc_manager = MultiprocessingManager(self.pool, worker_list, self.num_threads * batch_factor)
            ll = multiproc_manager.run()
        else:
            ll = [worker.run() for worker in worker_list]
        return 1.0/self.num_samples * np.sum(ll)

    def _get_gradient_log_lik(self, theta, batch_factor=2):
        """
        @param theta: the theta to calculate the likelihood for
        @param batch_factor: When using multiprocessing, we batch the samples together for speed
                            We make `num_threads` * `batch_factor` batches

        Calculate the gradient of the negative log likelihood - delegates to separate cpu threads if threads > 1
        """
        if self.pool is None:
            raise ValueError("Pool has not been initialized")

        exp_thetaT = np.exp(theta).T
        rand_seed = get_randint()
        worker_list = [
            GradientWorker(rand_seed + i, exp_thetaT, sample_data)
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
    def get_precalc_data(sample, feat_mut_steps, num_features):
        """
        @param sample: ImputedSequenceMutations

        Calculate the components in the gradient at the beginning of gradient descent
        Then the gradient can be calculated using element-wise matrix multiplication
        This is much faster than a for loop!

        We pre-calculate:
            1. `features_per_step_matrix`: the number of times each feature showed up at the mutation step
            2. `base_grad`: the gradient of the sum of the exp * psi terms
            3. `mutating_pos_feat_vals`: the feature idxs for which a mutation occured
        """
        mutating_pos_feat_vals = []
        base_grad = np.zeros((num_features, 1))
        # get the grad component from grad of psi * theta
        for feat_mut_step in feat_mut_steps:
            base_grad[feat_mut_step.mutating_pos_feat] += 1
            mutating_pos_feat_vals.append(feat_mut_step.mutating_pos_feat)

        # Get the grad component from grad of log(sum(exp(psi * theta)))
        # This matrix is just the number of times we saw each feature in the risk group
        features_per_step_matrix = np.zeros((
            num_features,
            sample.obs_seq_mutation.num_mutations
        ))
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

        mat_type = np.int8
        if np.max(np.abs(features_per_step_matrix)) > INT8_MAX:
            mat_type = np.int16 # range is -32768 to 32767

        return SamplePrecalcData(
            sample.obs_seq_mutation.feat_counts,
            csr_matrix(features_per_step_matrix.T, dtype=mat_type),
            base_grad,
            np.array(mutating_pos_feat_vals, dtype=np.int16),
        )

    @staticmethod
    def calculate_per_sample_log_lik(exp_theta, sample_data):
        """
        Calculate the log likelihood of this sample
        """
        risk_group_sum_base = np.dot(sample_data.init_feat_counts, exp_theta)
        risk_group_sum_deltas = sample_data.features_per_step_matrixT.dot(exp_theta)
        denominators = risk_group_sum_deltas + risk_group_sum_base
        numerators = exp_theta[sample_data.mutating_pos_feat_vals]
        log_lik = np.log(numerators).sum() - np.log(denominators).sum()
        return log_lik

    @staticmethod
    def get_gradient_log_lik_per_sample(exp_thetaT, sample_data):
        """
        Calculate the gradient of the log likelihood of this sample
        All the gradients for each step are the gradient of psi * theta - log(sum(exp(theta * psi)))
        Calculate the gradient from each step one at a time

        @param theta: the theta to evaluate the gradient at
        @param sample_data: SamplePrecalcData
        """
        features_per_step_matrixT = sample_data.features_per_step_matrixT.todense()
        grad_log_sum_exps_deltasT = np.multiply(features_per_step_matrixT, exp_thetaT)

        grad_log_sum_exps_baseT = np.multiply(sample_data.init_feat_counts, exp_thetaT)
        grad_log_sum_expsT = grad_log_sum_exps_baseT + grad_log_sum_exps_deltasT
        denominators = grad_log_sum_expsT.sum(axis=1)
        grad_components = np.divide(grad_log_sum_expsT, denominators)
        grad = sample_data.init_grad_vector.T - grad_components.sum(axis=0)
        return grad.T

class PrecalcDataWorker(ParallelWorker):
    """
    Stores the information for calculating gradient
    """
    def __init__(self, seed, sample, feat_mut_steps, num_features):
        """
        @param exp_theta: theta is where to take the gradient of the total log likelihood, exp_theta is exp(theta)
        @param sample: ImputedSequenceMutations
        @param feat_mut_steps: list of FeatureMutationStep
        @param num_features: total number of features that exist
        """
        self.seed = seed
        self.sample = sample
        self.feat_mut_steps = feat_mut_steps
        self.num_features = num_features

    def run_worker(self):
        """
        @return SamplePrecalcData
        """
        return SurvivalProblemCustom.get_precalc_data(self.sample, self.feat_mut_steps, self.num_features)

class GradientWorker(ParallelWorker):
    """
    Stores the information for calculating gradient
    """
    def __init__(self, seed, exp_thetaT, sample_data):
        """
        @param exp_thetaT: theta is where to take the gradient of the total log likelihood, exp_thetaT is exp(theta).T
        @param sample_data: class SamplePrecalcData
        """
        self.seed = seed
        self.exp_thetaT = exp_thetaT
        self.sample_data = sample_data

    def run_worker(self):
        """
        @return the gradient of the log likelihood for this sample
        """
        return SurvivalProblemCustom.get_gradient_log_lik_per_sample(self.exp_thetaT, self.sample_data)

class ObjectiveValueWorker(ParallelWorker):
    """
    Stores the information for calculating objective function value
    """
    def __init__(self, seed, exp_theta, sample_data):
        """
        @param exp_theta: theta is where to take the gradient of the total log likelihood, exp_theta is exp(theta)
        @param sample: SamplePrecalcData
        """
        self.seed = seed
        self.exp_theta = exp_theta
        self.sample_data = sample_data

    def run_worker(self):
        """
        @return the log likelihood for this sample
        """
        return SurvivalProblemCustom.calculate_per_sample_log_lik(self.exp_theta, self.sample_data)
