import time
from multiprocessing import Pool
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
    def __init__(self, init_feat_counts, features_per_step_matrixT, init_grad_vector, mutating_pos_feat_vals_rows, mutating_pos_feat_vals_cols, obs_seq_mutation, feat_mut_steps):
        self.init_feat_counts = init_feat_counts
        self.features_per_step_matrixT = features_per_step_matrixT
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

    def __init__(self, feat_generator, samples, penalty_params, per_target_model, theta_mask, fuse_windows=[], fuse_center_only=False, num_threads=1):
        self.feature_generator = feat_generator
        self.samples = samples
        self.theta_mask = theta_mask
        self.per_target_model = per_target_model
        self.num_samples = len(self.samples)
        self.penalty_params = penalty_params
        self.fuse_windows = fuse_windows
        self.fuse_center_only = fuse_center_only

        self.num_threads = num_threads
        self.pool = Pool(self.num_threads)
        self.precalc_data = self._create_precalc_data_parallel(samples)

        self.post_init()

    def post_init(self):
        return

    def close(self):
        """ Close pools if there is a pool open """
        # Ensure no more new jobs submitted to the pool
        self.pool.close()
        # Wait for the worker processes to exit -- make sure we don't keep these processes open!
        # Otherwise memory usage goes crazy
        self.pool.join()

    def solve(self):
        """
        Solve the problem and return the solution. Make sure to call self.pool.close()!!!
        """
        raise NotImplementedError()

    def _create_precalc_data_parallel(self, samples, batch_factor=4):
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
                self.per_target_model,
            ) for i, sample in enumerate(samples)
        ]
        if self.num_threads > 1:
            multiproc_manager = MultiprocessingManager(self.pool, worker_list, num_approx_batches=self.num_threads * batch_factor)
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
        if self.pool is None:
            raise ValueError("Pool has not been initialized")

        # exp_theta = np.exp(theta)
        rand_seed = get_randint()
        worker_list = [
            ObjectiveValueWorker(rand_seed + i, sample_data) for i, sample_data in enumerate(self.precalc_data)
        ]
        if self.num_threads > 1:
            multiproc_manager = MultiprocessingManager(
                self.pool,
                worker_list,
                shared_obj=theta,
                num_approx_batches=self.num_threads * batch_factor,
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
        if self.pool is None:
            raise ValueError("Pool has not been initialized")

        exp_thetaT = np.exp(theta).T
        rand_seed = get_randint()
        worker_list = [
            GradientWorker(rand_seed + i, sample_data) for i, sample_data in enumerate(self.precalc_data)
        ]
        if self.num_threads > 1:
            multiproc_manager = MultiprocessingManager(
                self.pool,
                worker_list,
                shared_obj=theta,
                num_approx_batches=self.num_threads * batch_factor,
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
        # This matrix is just the number of times we saw each feature in the risk group
        features_per_step_matrix = dok_matrix((
            num_features,
            sample.obs_seq_mutation.num_mutations
        ), dtype=np.int16)
        prev_feat_mut_step = feat_mut_steps[0]
        for i, feat_mut_step in enumerate(feat_mut_steps[1:]):
            nonzero_rows = features_per_step_matrix[:,i].nonzero()[0]
            if nonzero_rows.size:
                # All the features are very similar between risk groups - copy first
                features_per_step_matrix[nonzero_rows,i + 1] = features_per_step_matrix[nonzero_rows,i]

            # Remove feature corresponding to position that mutated already
            for f in prev_feat_mut_step.mutating_pos_feats:
                features_per_step_matrix[f, i + 1] -= 1

            # Need to update the terms for positions near the previous mutation
            # Remove old feature values
            old_feat_idxs = feat_mut_step.neighbors_feat_old.values()
            # it is possible to have the same feature idxs in this list - hence we need the for loop
            for f_list in old_feat_idxs:
                for f in f_list:
                    features_per_step_matrix[f, i + 1] -= 1

            # Add new feature values
            new_feat_idxs = feat_mut_step.neighbors_feat_new.values()
            # it is possible to have the same feature idxs in this list - hence we need the for loop
            for f_list in new_feat_idxs:
                for f in f_list:
                    features_per_step_matrix[f, i + 1] += 1

            prev_feat_mut_step = feat_mut_step

        mat_type = np.int8
        if np.max(np.abs(features_per_step_matrix)) > INT8_MAX:
            mat_type = np.int16 # range is -32768 to 32767

        return SamplePrecalcData(
            sample.obs_seq_mutation.feat_counts,
            csr_matrix(features_per_step_matrix.T, dtype=mat_type),
            base_grad,
            np.array(mutating_pos_feat_vals_rows, dtype=np.int16),
            np.array(mutating_pos_feat_vals_cols, dtype=np.int16),
            sample.obs_seq_mutation,
            feat_mut_steps,
        )

    @staticmethod
    def calculate_per_sample_log_lik(theta, sample_data, use_iterative=True):
        """
        Calculate the log likelihood of this sample
        """
        if use_iterative:
            # HACK: remove later
            denominators = [
                (np.exp(sample_data.obs_seq_mutation.feat_matrix_start * theta)).sum()
            ]
            prev_feat_mut_step = sample_data.feat_mut_steps[0]
            for i, feat_mut_step in enumerate(sample_data.feat_mut_steps[1:]):
                new_denom = SurvivalProblemCustom._get_denom_update(theta, denominators[i], prev_feat_mut_step.mutating_pos_feats, feat_mut_step)
                prev_feat_mut_step = feat_mut_step
                denominators.append(new_denom)

            numerators = theta[sample_data.mutating_pos_feat_vals_rows, sample_data.mutating_pos_feat_vals_cols]
            log_lik = numerators.sum() - np.log(denominators).sum()
        else:
            # Use dense matrix multiplication
            risk_group_sum_base = np.dot(sample_data.init_feat_counts, exp_theta)
            # Use sparse matrix multiplication
            risk_group_sum_deltas = sample_data.features_per_step_matrixT.dot(exp_theta)

            denominators = (risk_group_sum_deltas + risk_group_sum_base).sum(axis=1)
            numerators = exp_theta[sample_data.mutating_pos_feat_vals_rows, sample_data.mutating_pos_feat_vals_cols]
            log_lik = np.log(numerators).sum() - np.log(denominators).sum()
        return log_lik

    @staticmethod
    def _get_denom_update(theta, old_denominator, prev_feat_idxs, feat_mut_step):
        """
        Calculate the denominator of the next mutation step quickly by reusing past computations
        and incorporating the deltas appropriately

        @param old_denominator: the denominator from the previous mutation step
        @param old_log_numerator: the numerator from the previous mutation step
        @param feat_mut_step: the features that differed for this next mutation step
        """
        # CHECK AXIS!
        old_feat_theta_sums = [theta[feat_idxs].sum(axis=0) for feat_idxs in feat_mut_step.neighbors_feat_old.values()]
        new_feat_theta_sums = [theta[feat_idxs].sum(axis=0) for feat_idxs in feat_mut_step.neighbors_feat_new.values()]
        return old_denominator - np.exp(theta[prev_feat_idxs].sum(axis=0)).sum() - np.exp(old_feat_theta_sums).sum() + np.exp(new_feat_theta_sums).sum()

    @staticmethod
    def get_gradient_log_lik_per_sample(theta, sample_data, use_iterative=True):
        """
        Calculate the gradient of the log likelihood of this sample
        All the gradients for each step are the gradient of psi * theta - log(sum(exp(theta * psi)))
        Calculate the gradient from each step one at a time

        @param theta: the theta to evaluate the gradient at
        @param sample_data: SamplePrecalcData
        """
        if use_iterative:
            pos_exp_theta = np.exp(sample_data.obs_seq_mutation.feat_matrix_start * theta)
            denominator = pos_exp_theta.sum()
            risk_group_grad = sample_data.obs_seq_mutation.feat_matrix_start.T * pos_exp_theta

            risk_group_grads = [risk_group_grad/denominator]
            risk_group_grad_tot = risk_group_grad/denominator
            prev_denominator = denominator
            prev_feat_mut_step = sample_data.feat_mut_steps[0]
            for i, feat_mut_step in enumerate(sample_data.feat_mut_steps[1:]):
                new_risk_group_grad, new_denom = SurvivalProblemCustom._get_grad_update(
                    theta,
                    risk_group_grad,
                    prev_denominator,
                    prev_feat_mut_step.mutating_pos_feats,
                    feat_mut_step,
                )
                prev_feat_mut_step = feat_mut_step
                prev_denominator = new_denom
                risk_group_grad = new_risk_group_grad
                risk_group_grad_tot += new_risk_group_grad/new_denom
                risk_group_grads.append(new_risk_group_grad/new_denom)
            return sample_data.init_grad_vector - risk_group_grad_tot
        else:
            # Calculate the base gradient
            grad_log_sum_baseT = np.multiply(sample_data.init_feat_counts, exp_thetaT)
            features_per_step_matrixT = sample_data.features_per_step_matrixT.todense()

            if exp_thetaT.shape[0] == NUM_NUCLEOTIDES:
                grad_log_sum_expsTs = []
                for i in range(exp_thetaT.shape[0]):
                    grad_log_sum_exps_deltasT = np.multiply(features_per_step_matrixT, exp_thetaT[i,:])
                    grad_log_sum_expsTs.append(
                        grad_log_sum_baseT[i,:] + grad_log_sum_exps_deltasT
                    )
                grad_log_sum_expsT = np.hstack(grad_log_sum_expsTs)
                denominators = grad_log_sum_expsT.sum(axis=1)
                grad_components = np.divide(grad_log_sum_expsT, denominators)
                grad_components_sum = grad_components.sum(axis=0)
                grad_components_sum = np.reshape(grad_components_sum, (NUM_NUCLEOTIDES, grad_components_sum.size/NUM_NUCLEOTIDES)).T
                return sample_data.init_grad_vector - grad_components_sum
            else:
                # element-wise multiplication
                grad_log_sum_exps_deltasT = np.multiply(features_per_step_matrixT, exp_thetaT)
                grad_log_sum_expsT = grad_log_sum_baseT + grad_log_sum_exps_deltasT
                denominators = grad_log_sum_expsT.sum(axis=1)
                grad_components = np.divide(grad_log_sum_expsT, denominators)
                return sample_data.init_grad_vector - grad_components.sum(axis=0).T

    @staticmethod
    def _get_grad_update(theta, old_risk_group_grad, old_denominator, mutating_pos_feats, feat_mut_step):
        remove_term = np.exp(theta[mutating_pos_feats,:].sum(axis=0))
        new_denom = old_denominator - remove_term.sum()

        new_risk_group_grad = np.copy(old_risk_group_grad)
        new_risk_group_grad[mutating_pos_feats,:] -= remove_term

        for pos, feat_idxs in feat_mut_step.neighbors_feat_old.iteritems():
            remove_term = np.exp(theta[feat_idxs,:].sum(axis=0))
            new_risk_group_grad[feat_idxs,:] -= remove_term
            new_denom -= remove_term.sum()
        for pos, feat_idxs in feat_mut_step.neighbors_feat_new.iteritems():
            add_term = np.exp(theta[feat_idxs,:].sum(axis=0))
            new_risk_group_grad[feat_idxs,:] += add_term
            new_denom += add_term.sum()
        return new_risk_group_grad, new_denom

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
