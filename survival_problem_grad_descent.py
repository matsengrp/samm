import time
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix, dok_matrix

import logging as log
from survival_problem import SurvivalProblem
from parallel_worker import MultiprocessingManager, ParallelWorker
from common import *
from profile_support import profile

class SamplePrecalcData:
    """
    Stores data for gradient calculations
    """
    def __init__(self, features_per_step_matrices, features_sign_updates, init_grad_vector, mutating_pos_feat_vals_rows, mutating_pos_feat_vals_cols, obs_seq_mutation, feat_mut_steps):
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
    print_iter = 10 # print status every `print_iter` iterations

    def __init__(self, feat_generator, samples, sample_labels=None, penalty_params=[0], per_target_model=False, possible_theta_mask=None, zero_theta_mask=None, fuse_windows=[], fuse_center_only=False, pool=None):
        """
        @param sample_labels: only used for calculating the Hessian
        @param possible_theta_mask: these theta values are some finite number
        @param zero_theta_mask: these theta values are forced to be zero
        """
        self.feature_generator = feat_generator
        self.samples = samples
        self.possible_theta_mask = possible_theta_mask
        self.zero_theta_mask = zero_theta_mask
        if zero_theta_mask is not None and possible_theta_mask is not None:
            self.theta_mask_flat = (possible_theta_mask & ~zero_theta_mask).reshape((zero_theta_mask.size,), order="F")

        self.num_samples = len(self.samples)
        self.sample_labels = sample_labels
        if self.sample_labels is not None:
            assert(len(self.sample_labels) == self.num_samples)
            self.num_reps_per_obs = self.num_samples/len(set(sample_labels))

        self.per_target_model = per_target_model
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

    def _group_log_lik_ratio_vec(self, ll_ratio_vec):
        num_unique_samples = len(set(self.sample_labels))
        # Reshape the log likelihood ratio vector
        ll_ratio_dict = [[] for _ in range(num_unique_samples)]
        for v, label in zip(ll_ratio_vec.tolist(), self.sample_labels):
            ll_ratio_dict[label].append(v)
        ll_ratio_reshape = (np.array(ll_ratio_dict)).T
        ll_ratio_vec_sums = ll_ratio_reshape.sum(axis=1)/num_unique_samples
        return ll_ratio_vec_sums

    def calculate_log_lik_ratio_vec(self, theta1, theta2, group_by_sample=False):
        """
        @param theta: the theta in the numerator
        @param prev_theta: the theta in the denominator
        @return the log likelihood ratios between theta and prev_theta for each e-step sample
        """
        ll_vec1 = self._get_log_lik_parallel(theta1)
        ll_vec2 = self._get_log_lik_parallel(theta2)
        ll_ratio_vec = ll_vec1 - ll_vec2
        if group_by_sample:
            return self._group_log_lik_ratio_vec(ll_ratio_vec)
        else:
            return ll_ratio_vec

    def _get_log_lik_parallel(self, theta, batch_factor=4):
        """
        JUST KIDDING - parallel is not faster
        @param theta: the theta to calculate the likelihood for
        @param batch_factor: When using multiprocessing, we batch the samples together for speed
                            We make `num_threads` * `batch_factor` batches
        @return vector of log likelihood values
        """
        def _get_parallel(worker_list, shared_obj):
            if False and self.pool is not None and len(worker_list) > 10000:
                multiproc_manager = MultiprocessingManager(self.pool, worker_list, shared_obj=shared_obj, num_approx_batches=self.pool._processes * batch_factor)
                res = multiproc_manager.run()
            else:
                res = [worker.run(shared_obj) for worker in worker_list]
            return np.array(res)

        rand_seed = get_randint()
        worker_list = [
            ObjectiveValueWorker(rand_seed + i, sample_data, self.per_target_model) for i, sample_data in enumerate(self.precalc_data)
        ]
        return _get_parallel(worker_list, theta)

    def get_hessian(self, theta, batch_factor=4):
        """
        Uses Louis's method to calculate the information matrix of the observed data
        IMPORTANT: all the parallel workers that produce square matrices pre-batch jobs! We do this because otherwise memory consumption will go crazy.

        @return fishers information matrix of the observed data, hessian of the log likelihood of the complete data
        """
        def _get_parallel_sum(worker_list, shared_obj):
            if self.pool is not None:
                multiproc_manager = MultiprocessingManager(self.pool, worker_list, shared_obj=shared_obj, num_approx_batches=self.pool._processes * batch_factor)
                res = multiproc_manager.run()
            else:
                res = [worker.run(shared_obj) for worker in worker_list]
            tot = 0
            for r in res:
                tot += r
            return tot

        st = time.time()
        rand_seed = get_randint()
        worker_list = [
            GradientWorker(rand_seed + i, sample_data, self.per_target_model) for i, sample_data in enumerate(self.precalc_data)
        ]
        grad_log_lik = [worker.run(theta) for worker in worker_list]
        sorted_sample_labels = sorted(list(set(self.sample_labels)))
        log.info("Obtained gradients %s" % (time.time() - st))

        # Get the expected scores and their sum
        expected_scores = {label: 0 for label in sorted_sample_labels}
        for g, sample_label in zip(grad_log_lik, self.sample_labels):
            g = g.reshape((g.size, 1), order="F")
            expected_scores[sample_label] += g

        expected_scores_sum = 0
        for sample_label in sorted_sample_labels:
            expected_scores_sum += expected_scores[sample_label]
        log.info("Obtained expected scores %s" % (time.time() - st))

        # Calculate the score score (second summand)
        num_batches = self.pool._processes * batch_factor * 2 if self.pool is not None else 1
        batched_idxs = get_batched_list(range(len(grad_log_lik)), num_batches)
        score_score_worker_list = [
            ScoreScoreWorker(rand_seed + i, [grad_log_lik[j] for j in idxs]) for i, idxs in enumerate(batched_idxs)
        ]
        tot_score_score = _get_parallel_sum(score_score_worker_list, None)
        log.info("Obtained score scores %s" % (time.time() - st))

        # Calculate the cross scores (third summand)
        # Instead of calculating \sum_{i \neq j} ES_i ES_j^T directly we calculate
        # (\sum_{i} ES_i) ( \sum_{i} ES_i)^T - \sum_{i} ES_i ES_i^T
        batched_labels = get_batched_list(sorted_sample_labels, num_batches)
        expected_score_worker_list = [ExpectedScoreScoreWorker(rand_seed + i, labels) for i, labels in enumerate(batched_labels)]
        tot_expected_score_score = _get_parallel_sum(expected_score_worker_list, expected_scores)
        tot_cross_expected_scores = expected_scores_sum * expected_scores_sum.T - tot_expected_score_score
        log.info("Obtained cross scores %s" % (time.time() - st))

        # Calculate the hessian (first summand)
        assert(len(self.precalc_data) == len(grad_log_lik))
        hessian_worker_list = [
            HessianWorker(rand_seed + i, [self.precalc_data[j] for j in idxs], self.per_target_model) for i, idxs in enumerate(batched_idxs)
        ]
        hessian_sum = _get_parallel_sum(hessian_worker_list, theta)
        log.info("Obtained Hessian %s" % (time.time() - st))

        fisher_info = 1.0/self.num_reps_per_obs * (- hessian_sum - tot_score_score) - np.power(self.num_reps_per_obs, -2.0) * tot_cross_expected_scores
        return fisher_info, -1.0/self.num_samples * hessian_sum

    def _get_gradient_log_lik(self, theta, batch_factor=4):
        """
        JUST KIDDING - parallel is not faster
        @param theta: the theta to calculate the likelihood for
        @param batch_factor: When using multiprocessing, we batch the samples together for speed
                            We make `num_threads` * `batch_factor` batches

        Calculate the gradient of the negative log likelihood - delegates to separate cpu threads if threads > 1
        """
        def _get_parallel(worker_list, shared_obj):
            if False and self.pool is not None and len(worker_list) > 10000:
                multiproc_manager = MultiprocessingManager(self.pool, worker_list, shared_obj=shared_obj, num_approx_batches=self.pool._processes * batch_factor)
                res = multiproc_manager.run()
            else:
                res = [worker.run(shared_obj) for worker in worker_list]
            return res

        rand_seed = get_randint()
        worker_list = [
            GradientWorker(rand_seed + i, sample_data, self.per_target_model) for i, sample_data in enumerate(self.precalc_data)
        ]
        grad_ll_raw = _get_parallel(worker_list, theta)
        grad_ll_dtheta = np.sum(grad_ll_raw, axis=0)

        # Zero out all gradients that affect the constant theta values.
        if self.zero_theta_mask is not None:
            grad_ll_dtheta[self.zero_theta_mask] = 0

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
        num_targets = NUM_NUCLEOTIDES + 1 if per_target_model else 1
        base_grad = np.zeros((num_features, num_targets))
        # get the grad component from grad of psi * theta
        for i, feat_mut_step in enumerate(feat_mut_steps):
            base_grad[feat_mut_step.mutating_pos_feats, 0] += 1
            if per_target_model:
                col_idx = get_target_col(sample.obs_seq_mutation, sample.mutation_order[i])
                base_grad[feat_mut_step.mutating_pos_feats, col_idx] += 1
            mutating_pos_feat_vals_rows = np.append(mutating_pos_feat_vals_rows, feat_mut_step.mutating_pos_feats)
            if per_target_model:
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

            features_per_step_matrices.append(csr_matrix(pos_feat_matrix, dtype=np.int8))
            features_sign_updates.append(
                np.reshape(np.concatenate([-1 * np.ones(num_old + 1), np.ones(num_new)]), (num_old + num_new + 1, 1))
            )

            prev_feat_mut_step = feat_mut_step

        return SamplePrecalcData(
            features_per_step_matrices,
            features_sign_updates,
            base_grad,
            np.array(mutating_pos_feat_vals_rows, dtype=np.int16),
            np.array(mutating_pos_feat_vals_cols, dtype=np.int16),
            sample.obs_seq_mutation,
            feat_mut_steps,
        )

    @staticmethod
    def calculate_per_sample_log_lik(theta, sample_data, per_target_model):
        """
        Calculate the log likelihood of this sample
        """
        merged_thetas = theta[:,0, None]
        if per_target_model:
            merged_thetas = merged_thetas + theta[:,1:]
        prev_denom = (np.exp(sample_data.obs_seq_mutation.feat_matrix_start.dot(merged_thetas))).sum()
        denominators = [prev_denom]
        for pos_feat_matrix, features_sign_update in zip(sample_data.features_per_step_matrices, sample_data.features_sign_updates):
            exp_thetas = np.exp(pos_feat_matrix.dot(merged_thetas))
            signed_exp_thetas = np.multiply(exp_thetas, features_sign_update)
            new_denom = prev_denom + signed_exp_thetas.sum()
            denominators.append(new_denom)
            prev_denom = new_denom

        numerators = theta[sample_data.mutating_pos_feat_vals_rows, 0]
        if per_target_model:
            numerators = numerators + theta[sample_data.mutating_pos_feat_vals_rows, sample_data.mutating_pos_feat_vals_cols]

        log_lik = numerators.sum() - np.log(denominators).sum()
        return log_lik

    @staticmethod
    def get_gradient_log_lik_per_sample(theta, sample_data, per_target_model):
        """
        Calculate the gradient of the log likelihood of this sample
        All the gradients for each step are the gradient of psi * theta - log(sum(exp(theta * psi)))
        Calculate the gradient from each step one at a time

        @param theta: the theta to evaluate the gradient at
        @param sample_data: SamplePrecalcData
        """
        merged_thetas = theta[:,0, None]
        if per_target_model:
            merged_thetas = merged_thetas + theta[:,1:]
        pos_exp_theta = np.exp(sample_data.obs_seq_mutation.feat_matrix_start.dot(merged_thetas))
        prev_denom = pos_exp_theta.sum()

        prev_risk_group_grad = sample_data.obs_seq_mutation.feat_matrix_start.transpose().dot(pos_exp_theta)

        risk_group_grad_tot = prev_risk_group_grad/prev_denom
        for pos_feat_matrix, pos_feat_matrixT, features_sign_update in zip(sample_data.features_per_step_matrices, sample_data.features_per_step_matricesT, sample_data.features_sign_updates):
            exp_thetas = np.exp(pos_feat_matrix.dot(merged_thetas))
            signed_exp_thetas = np.multiply(exp_thetas, features_sign_update)

            prev_risk_group_grad += pos_feat_matrixT.dot(signed_exp_thetas)

            prev_denom += signed_exp_thetas.sum()
            prev_denom_inv = 1.0/prev_denom
            risk_group_grad_tot += prev_risk_group_grad * prev_denom_inv
        if per_target_model:
            risk_group_grad_tot = np.hstack([np.sum(risk_group_grad_tot, axis=1, keepdims=True), risk_group_grad_tot])
        return sample_data.init_grad_vector - risk_group_grad_tot

    @staticmethod
    def get_hessian_per_sample(theta, sample_data, per_target_model):
        """
        Calculates the second derivative of the log likelihood for the complete data
        """
        merged_thetas = theta[:,0, None]
        if per_target_model:
            merged_thetas = merged_thetas + theta[:,1:]

        # Create base dd matrix.
        dd_matrices = [np.zeros((theta.shape[0], theta.shape[0])) for i in range(theta.shape[1])]
        for pos in range(sample_data.obs_seq_mutation.seq_len):
            exp_theta_psi = np.exp(np.array(sample_data.obs_seq_mutation.feat_matrix_start[pos,:] * merged_thetas).flatten())
            features = sample_data.obs_seq_mutation.feat_matrix_start[pos,:].nonzero()[1]
            for f1 in features:
                for f2 in features:
                    # The first column in a per-target model appears in all other columns. Hence we need a sum of all the exp_thetas
                    dd_matrices[0][f1, f2] += exp_theta_psi.sum()
                    for j in range(1, theta.shape[1]):
                        # The rest of the theta values for the per-target model only appear once in the exp_theta vector
                        dd_matrices[j][f1, f2] += exp_theta_psi[j - 1]

        pos_exp_theta = np.exp(sample_data.obs_seq_mutation.feat_matrix_start.dot(merged_thetas))
        prev_denom = pos_exp_theta.sum()

        prev_risk_group_grad = sample_data.obs_seq_mutation.feat_matrix_start.transpose().dot(pos_exp_theta)
        if per_target_model:
            # Deal with the fact that the first column is special in a per-target model
            aug_prev_risk_group_grad = np.hstack([np.sum(prev_risk_group_grad, axis=1, keepdims=True), prev_risk_group_grad])
            aug_prev_risk_group_grad = aug_prev_risk_group_grad.reshape((aug_prev_risk_group_grad.size, 1), order="F")
        else:
            aug_prev_risk_group_grad = prev_risk_group_grad.reshape((prev_risk_group_grad.size, 1), order="F")

        block_diag_dd = sp.linalg.block_diag(*dd_matrices)
        for i in range(theta.shape[1] - 1):
            # Recall that the first column in a per-target model appears in all the exp_theta expressions. So we need to add in these values
            # to the off-diagonal blocks of the second-derivative matrix
            block_diag_dd[(i + 1) * theta.shape[0]:(i + 2) * theta.shape[0], 0:theta.shape[0]] = dd_matrices[i + 1]
            block_diag_dd[0:theta.shape[0], (i + 1) * theta.shape[0]:(i + 2) * theta.shape[0]] = dd_matrices[i + 1]

        risk_group_hessian = aug_prev_risk_group_grad * aug_prev_risk_group_grad.T * np.power(prev_denom, -2) - np.power(prev_denom, -1) * block_diag_dd
        for pos_feat_matrix, pos_feat_matrixT, features_sign_update in zip(sample_data.features_per_step_matrices, sample_data.features_per_step_matricesT, sample_data.features_sign_updates):
            exp_thetas = np.exp(pos_feat_matrix.dot(merged_thetas))
            signed_exp_thetas = np.multiply(exp_thetas, features_sign_update)

            prev_risk_group_grad += pos_feat_matrixT.dot(signed_exp_thetas)

            prev_denom += signed_exp_thetas.sum()

            # Now update the dd_matrix after the previous mutation step
            for i in range(pos_feat_matrix.shape[0]):
                feature_vals = pos_feat_matrix[i,:].nonzero()[1]
                for f1 in feature_vals:
                    for f2 in feature_vals:
                        dd_matrices[0][f1, f2] += signed_exp_thetas[i,:].sum()
                        for j in range(1, theta.shape[1]):
                            dd_matrices[j][f1, f2] += signed_exp_thetas[i,j - 1]

            if per_target_model:
                aug_prev_risk_group_grad = np.hstack([np.sum(prev_risk_group_grad, axis=1, keepdims=True), prev_risk_group_grad])
                aug_prev_risk_group_grad = aug_prev_risk_group_grad.reshape((aug_prev_risk_group_grad.size, 1), order="F")
            else:
                aug_prev_risk_group_grad = prev_risk_group_grad.reshape((prev_risk_group_grad.size, 1), order="F")

            block_diag_dd = sp.linalg.block_diag(*dd_matrices)
            for i in range(theta.shape[1] - 1):
                block_diag_dd[(i + 1) * theta.shape[0]:(i + 2) * theta.shape[0], 0:theta.shape[0]] = dd_matrices[i + 1]
                block_diag_dd[0:theta.shape[0], (i + 1) * theta.shape[0]:(i + 2) * theta.shape[0]] = dd_matrices[i + 1]

            risk_group_hessian += aug_prev_risk_group_grad * aug_prev_risk_group_grad.T * np.power(prev_denom, -2) - np.power(prev_denom, -1) * block_diag_dd
        return risk_group_hessian

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
    def __init__(self, seed, sample_data, per_target_model):
        """
        @param sample_data: class SamplePrecalcData
        """
        self.seed = seed
        self.sample_data = sample_data
        self.per_target_model = per_target_model

    def run_worker(self, theta):
        """
        @param exp_thetaT: theta is where to take the gradient of the total log likelihood, exp_thetaT is exp(theta).T
        @return the gradient of the log likelihood for this sample
        """
        return SurvivalProblemCustom.get_gradient_log_lik_per_sample(theta, self.sample_data, self.per_target_model)

    def __str__(self):
        return "GradientWorker %s" % self.sample_data.obs_seq_mutation

class ObjectiveValueWorker(ParallelWorker):
    """
    Stores the information for calculating objective function value
    """
    def __init__(self, seed, sample_data, per_target_model):
        """
        @param sample: SamplePrecalcData
        """
        self.seed = seed
        self.sample_data = sample_data
        self.per_target_model = per_target_model

    def run_worker(self, theta):
        """
        @param exp_theta: the exp of theta
        @return the log likelihood for this sample
        """
        return SurvivalProblemCustom.calculate_per_sample_log_lik(theta, self.sample_data, self.per_target_model)

    def __str__(self):
        return "ObjectiveValueWorker %s" % self.sample_data.obs_seq_mutation

class HessianWorker(ParallelWorker):
    """
    Stores the information for calculating gradient
    """
    def __init__(self, seed, sample_datas, per_target_model):
        """
        @param sample_data: class SamplePrecalcData
        """
        self.seed = seed
        self.sample_datas = sample_datas
        self.per_target_model = per_target_model

    def run_worker(self, theta):
        """
        @param exp_thetaT: theta is where to take the gradient of the total log likelihood, exp_thetaT is exp(theta).T
        @return the gradient of the log likelihood for this sample
        """
        tot_hessian = 0
        for s in self.sample_datas:
            h = SurvivalProblemCustom.get_hessian_per_sample(theta, s, self.per_target_model)
            tot_hessian += h
        return tot_hessian

class ScoreScoreWorker(ParallelWorker):
    """
    Calculate the product of scores
    """
    def __init__(self, seed, grad_log_liks):
        """
        @param grad_log_liks: the grad_log_liks to calculate the product of scores
        """
        self.seed = seed
        self.grad_log_liks = grad_log_liks

    def run_worker(self, shared_obj=None):
        """
        @return the sum of the product of scores
        """
        ss = 0
        for g in self.grad_log_liks:
            g = g.reshape((g.size, 1), order="F")
            ss += g * g.T
        return ss

class ExpectedScoreScoreWorker(ParallelWorker):
    """
    Calculate the product of expected scores between itself (indexed by labels)
    """
    def __init__(self, seed, label_list):
        """
        @param label_list: a list of labels to process
        """
        self.seed = seed
        self.label_list = label_list

    def run_worker(self, expected_scores):
        """
        @param expected_scores: the expected scores
        @return the sum of the product of expected scores for the given label_list
        """
        ss = 0
        for label in self.label_list:
            ss += expected_scores[label] * expected_scores[label].T
        return ss
