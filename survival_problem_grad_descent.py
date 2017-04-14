import time
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix, dok_matrix

import logging as log
from survival_problem import SurvivalProblem
from submotif_feature_generator import SubmotifFeatureGenerator
from parallel_worker import MultiprocessingManager, ParallelWorker
from common import *
from profile_support import profile

class SamplePrecalcData:
    """
    Stores data for gradient calculations
    """
    def __init__(self, init_feat_counts, features_per_step_matrices, features_sign_updates, init_grad_vector, mutating_pos_feat_vals_rows, mutating_pos_feat_vals_cols, obs_seq_mutation, feat_mut_steps):
        self.init_feat_counts = init_feat_counts
        self.features_per_step_matrices = features_per_step_matrices
        self.features_per_step_matricesT = [m.transpose() for m in features_per_step_matrices]
        self.features_sign_updates = features_sign_updates
        self.init_grad_vector = init_grad_vector
        self.mutating_pos_feat_vals_rows = mutating_pos_feat_vals_rows
        self.mutating_pos_feat_vals_cols = mutating_pos_feat_vals_cols
        self.obs_seq_mutation = obs_seq_mutation
        self.feat_mut_steps = feat_mut_steps

class SurvivalProblemCustom(SurvivalProblem):
    """
    Our own implementation to solve the survival problem
    """
    print_iter = 5 # print status every `print_iter` iterations

    def __init__(self, feat_generator, samples, penalty_params, per_target_model, theta_mask, fuse_windows=[], fuse_center_only=False, pool=None):
        self.feature_generator = feat_generator
        self.samples = samples
        self.theta_mask = theta_mask
        self.per_target_model = per_target_model
        self.num_samples = len(self.samples)
        self.penalty_params = penalty_params
        self.fuse_windows = fuse_windows
        self.fuse_center_only = fuse_center_only

        self.pool = pool

        self.precalc_data = self._create_precalc_data_parallel(samples)

        self.post_init()

    def post_init(self):
        return

    def solve(self):
        """
        Solve the problem and return the solution. Make sure to call self.pool.close()!!!
        """
        raise NotImplementedError()

    def _create_precalc_data_parallel(self, samples, batch_factor=4):
        """
        calculate the precalculated data for each sample in parallel
        """
        rand_seed = get_randint()
        worker_list = [
            PrecalcDataWorker(
                rand_seed + i,
                sample,
                self.feature_generator.create_for_mutation_steps(sample),
                self.feature_generator.feature_vec_len,
                self.per_target_model,
            ) for i, sample in enumerate(samples)
        ]
        if self.pool is not None:
            multiproc_manager = MultiprocessingManager(self.pool, worker_list, num_approx_batches=self.pool._processes * batch_factor)
            precalc_data = multiproc_manager.run()
        else:
            precalc_data = [worker.run(None) for worker in worker_list]
        return precalc_data

    def get_value(self, theta):
        """
        @return negative penalized log likelihood
        """
        raise NotImplementedError()

    def calculate_log_lik_ratio_vec(self, theta1, theta2):
        """
        @param theta: the theta in the numerator
        @param prev_theta: the theta in the denominator
        @return the log likelihood ratios between theta and prev_theta for each e-step sample
        """
        ll_vec1 = self._get_log_lik_parallel(theta1)
        ll_vec2 = self._get_log_lik_parallel(theta2)
        return ll_vec1 - ll_vec2


    def _get_log_lik_parallel(self, theta, batch_factor=2):
        """
        @param theta: the theta to calculate the likelihood for
        @param batch_factor: When using multiprocessing, we batch the samples together for speed
                            We make `num_threads` * `batch_factor` batches
        @return vector of log likelihood values
        """
        rand_seed = get_randint()
        worker_list = [
            ObjectiveValueWorker(rand_seed + i, sample_data) for i, sample_data in enumerate(self.precalc_data)
        ]
        if self.pool is not None:
            multiproc_manager = MultiprocessingManager(
                self.pool,
                worker_list,
                shared_obj=theta,
                num_approx_batches=self.pool._processes * batch_factor,
            )
            ll = multiproc_manager.run()
        else:
            ll = [worker.run(theta) for worker in worker_list]
        return np.array(ll)

    def _get_gradient_log_lik(self, theta, batch_factor=2):
        """
        @param theta: the theta to calculate the likelihood for
        @param batch_factor: When using multiprocessing, we batch the samples together for speed
                            We make `num_threads` * `batch_factor` batches

        Calculate the gradient of the negative log likelihood - delegates to separate cpu threads if threads > 1
        """
        rand_seed = get_randint()
        worker_list = [
            GradientWorker(rand_seed + i, sample_data) for i, sample_data in enumerate(self.precalc_data)
        ]
        if self.pool is not None:
            multiproc_manager = MultiprocessingManager(
                self.pool,
                worker_list,
                shared_obj=theta,
                num_approx_batches=self.pool._processes * batch_factor,
            )
            l = multiproc_manager.run()
        else:
            l = [worker.run(theta) for worker in worker_list]

        grad_ll_dtheta = np.sum(l, axis=0)
        return -1.0/self.num_samples * grad_ll_dtheta

    @staticmethod
    def get_precalc_data(sample, feat_mut_steps, num_features, per_target_model):
        """
        @param sample: ImputedSequenceMutations
        @param feat_mut_steps: list of FeatureMutationStep
        @param num_features: total number of features
        @param per_target_model: True if estimating different hazards for different target nucleotides

        Calculate the components in the gradient at the beginning of gradient descent
        Then the gradient can be calculated using element-wise matrix multiplication
        This is much faster than a for loop!

        We pre-calculate:
            1. `features_per_step_matrix`: the number of times each feature showed up at the mutation step
            2. `base_grad`: the gradient of the sum of the exp * psi terms
            3. `mutating_pos_feat_vals_rows`: the feature row idxs for which a mutation occured
            4. `mutating_pos_feat_vals_cols`: the feature column idxs for which a mutation occured
        """
        mutating_pos_feat_vals_rows = np.array([])
        mutating_pos_feat_vals_cols = np.array([])
        num_targets = NUM_NUCLEOTIDES if per_target_model else 1
        base_grad = np.zeros((num_features, num_targets))
        # get the grad component from grad of psi * theta
        for i, feat_mut_step in enumerate(feat_mut_steps):
            col_idx = get_target_col(sample.obs_seq_mutation, sample.mutation_order[i]) if per_target_model else 0
            base_grad[feat_mut_step.mutating_pos_feats, col_idx] += 1
            mutating_pos_feat_vals_rows = np.append(mutating_pos_feat_vals_rows, feat_mut_step.mutating_pos_feats)
            mutating_pos_feat_vals_cols = np.append(mutating_pos_feat_vals_cols, [col_idx] * len(feat_mut_step.mutating_pos_feats))

        # Get the grad component from grad of log(sum(exp(psi * theta)))
        features_per_step_matrices = []
        features_sign_updates = []
        prev_feat_mut_step = feat_mut_steps[0]
        for i, feat_mut_step in enumerate(feat_mut_steps[1:]):
            num_old = len(feat_mut_step.neighbors_feat_old)
            num_new = len(feat_mut_step.neighbors_feat_new)
            # First row of this matrix is the mutating pos. Next set are the positions with their old feature idxs. Then all the positions with their new feature idxs.
            # plus one because one position mutates and must be removed
            pos_feat_matrix = dok_matrix((
                num_old + num_new + 1,
                num_features,
            ), dtype=np.int8)

            # Remove feature corresponding to position that mutated already
            pos_feat_matrix[0, prev_feat_mut_step.mutating_pos_feats] = 1

            # Need to update the terms for positions near the previous mutation
            # Remove old feature values
            old_feat_idxs = feat_mut_step.neighbors_feat_old.values()
            for f_idx, f_list in enumerate(old_feat_idxs):
                pos_feat_matrix[f_idx + 1, f_list] = 1

            # Add new feature values
            new_feat_idxs = feat_mut_step.neighbors_feat_new.values()
            for f_idx, f_list in enumerate(new_feat_idxs):
                pos_feat_matrix[f_idx + 1 + num_old, f_list] = 1

            features_per_step_matrices.append(csr_matrix(pos_feat_matrix))
            features_sign_updates.append(
                np.reshape(np.concatenate([-1 * np.ones(num_old + 1), np.ones(num_new)]), (num_old + num_new + 1, 1))
            )

            prev_feat_mut_step = feat_mut_step

        return SamplePrecalcData(
            sample.obs_seq_mutation.feat_counts,
            features_per_step_matrices,
            features_sign_updates,
            base_grad,
            np.array(mutating_pos_feat_vals_rows, dtype=np.int16),
            np.array(mutating_pos_feat_vals_cols, dtype=np.int16),
            sample.obs_seq_mutation,
            feat_mut_steps,
        )

    @staticmethod
    def calculate_per_sample_log_lik(theta, sample_data):
        """
        Calculate the log likelihood of this sample
        """
        denominators = [
            (np.exp(sample_data.obs_seq_mutation.feat_matrix_start * theta)).sum()
        ]
        prev_denom = (np.exp(sample_data.obs_seq_mutation.feat_matrix_start * theta)).sum()
        for pos_feat_matrix, features_sign_update in zip(sample_data.features_per_step_matrices, sample_data.features_sign_updates):
            exp_thetas = np.exp(pos_feat_matrix.dot(theta))
            signed_exp_thetas = np.multiply(exp_thetas, features_sign_update)
            new_denom = prev_denom + signed_exp_thetas.sum()
            denominators.append(new_denom)
            prev_denom = new_denom

        numerators = theta[sample_data.mutating_pos_feat_vals_rows, sample_data.mutating_pos_feat_vals_cols]
        log_lik = numerators.sum() - np.log(denominators).sum()
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
        denominators = [
            (np.exp(sample_data.obs_seq_mutation.feat_matrix_start * theta)).sum()
        ]
        prev_denom = (np.exp(sample_data.obs_seq_mutation.feat_matrix_start * theta)).sum()

        pos_exp_theta = np.exp(sample_data.obs_seq_mutation.feat_matrix_start * theta)
        denominator = pos_exp_theta.sum()
        prev_risk_group_grad = sample_data.obs_seq_mutation.feat_matrix_start.T * pos_exp_theta

        risk_group_grads = [prev_risk_group_grad/denominator]
        risk_group_grad_tot = prev_risk_group_grad/denominator
        prev_denominator = denominator
        for pos_feat_matrix, pos_feat_matrixT, features_sign_update in zip(sample_data.features_per_step_matrices, sample_data.features_per_step_matricesT, sample_data.features_sign_updates):
            exp_thetas = np.exp(pos_feat_matrix.dot(theta))
            signed_exp_thetas = np.multiply(exp_thetas, features_sign_update)

            new_denom = prev_denom + signed_exp_thetas.sum()
            denominators.append(new_denom)
            grad_update = pos_feat_matrixT.dot(signed_exp_thetas)
            risk_group_grad = prev_risk_group_grad + grad_update
            risk_group_grads.append(risk_group_grad/new_denom)
            risk_group_grad_tot += risk_group_grad/new_denom

            prev_denom = new_denom
            prev_risk_group_grad = risk_group_grad
        return sample_data.init_grad_vector - risk_group_grad_tot

class PrecalcDataWorker(ParallelWorker):
    """
    Stores the information for calculating gradient
    """
    def __init__(self, seed, sample, feat_mut_steps, num_features, per_target_model):
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
        self.per_target_model = per_target_model

    def run_worker(self, shared_obj=None):
        """
        @param shared_obj: ignored object
        @return SamplePrecalcData
        """
        return SurvivalProblemCustom.get_precalc_data(self.sample, self.feat_mut_steps, self.num_features, self.per_target_model)

class GradientWorker(ParallelWorker):
    """
    Stores the information for calculating gradient
    """
    def __init__(self, seed, sample_data):
        """
        @param sample_data: class SamplePrecalcData
        """
        self.seed = seed
        self.sample_data = sample_data

    def run_worker(self, theta):
        """
        @param exp_thetaT: theta is where to take the gradient of the total log likelihood, exp_thetaT is exp(theta).T
        @return the gradient of the log likelihood for this sample
        """
        return SurvivalProblemCustom.get_gradient_log_lik_per_sample(theta, self.sample_data)

    def __str__(self):
        return "GradientWorker %s" % self.sample_data.init_feat_counts

class ObjectiveValueWorker(ParallelWorker):
    """
    Stores the information for calculating objective function value
    """
    def __init__(self, seed, sample_data):
        """
        @param sample: SamplePrecalcData
        """
        self.seed = seed
        self.sample_data = sample_data

    def run_worker(self, theta):
        """
        @param exp_theta: the exp of theta
        @return the log likelihood for this sample
        """
        return SurvivalProblemCustom.calculate_per_sample_log_lik(theta, self.sample_data)

    def __str__(self):
        return "ObjectiveValueWorker %s" % self.sample_data.init_feat_counts
