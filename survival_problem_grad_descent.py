import time
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix, dok_matrix

import logging as log
from combined_feature_generator import CombinedFeatureGenerator
from survival_problem import SurvivalProblem
from survival_problem_grad_descent_workers import *
from common import *
from parallel_worker import MultiprocessingManager
from profile_support import profile

class SurvivalProblemCustom(SurvivalProblem):
    """
    Our own implementation to solve the survival problem
    """
    print_iter = 10 # print status every `print_iter` iterations

    def __init__(self, feat_generator, samples, sample_labels=None, penalty_params=[0], per_target_model=False, possible_theta_mask=None, zero_theta_mask=None, fuse_windows=[], fuse_center_only=False, pool=None):
        """
        @param feat_generator: CombinedFeatureGenerator
        @param samples: observations to compute gradient descent problem
        @param sample_labels: only used for calculating the Hessian
        @param possible_theta_mask: these theta values are some finite number
        @param zero_theta_mask: these theta values are forced to be zero
        """
        assert(isinstance(feat_generator, CombinedFeatureGenerator))
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

    def _create_precalc_data_parallel(self, samples):
        """
        calculate the precalculated data for each sample in parallel
        """
        st = time.time()
        worker_list = [
            PrecalcDataWorker(
                sample,
                self.feature_generator.create_for_mutation_steps(sample),
                self.feature_generator.feature_vec_len,
                self.per_target_model,
            ) for sample in samples
        ]
        precalc_data = self._run_processes(worker_list, shared_obj=None, pool=self.pool)
        log.info("precalc time %f", time.time() - st)
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

    def _run_processes(self, worker_list, shared_obj=None, pool=None):
        """
        Run and join python Thread objects

        @param worker_list: a list of Worker objects from down there
        """
        if pool is not None:
            manager = MultiprocessingManager(
                    pool,
                    worker_list,
                    shared_obj=shared_obj,
                    num_approx_batches=pool._processes * 2)
            results = manager.run()
        else:
            results = [w.run(shared_obj) for w in worker_list]
        return results

    def _get_log_lik_parallel(self, theta):
        """
        NOTE: parallel not faster if not a lot of a data

        @param theta: the theta to calculate the likelihood for
        @return vector of log likelihood values
        """
        worker_list = [
            LogLikelihoodWorker(s, self.per_target_model)
            for s in self.precalc_data
        ]
        log_liks = self._run_processes(
                worker_list,
                shared_obj=theta,
                pool=self.pool if len(worker_list) > 10000 else None)

        return np.array(log_liks)

    def get_hessian(self, theta):
        """
        Uses Louis's method to calculate the information matrix of the observed data

        @return fishers information matrix of the observed data, hessian of the log likelihood of the complete data
        """
        def _get_parallel_sum(worker_list, shared_obj=None):
            res_list = self._run_processes(worker_list, shared_obj, pool=self.pool)
            assert not any([r is None for r in res_list])
            tot = 0
            for r in res_list:
                tot += r
            return tot

        st = time.time()
        grad_worker_list = [
            GradientWorker([s], self.per_target_model) for s in self.precalc_data
        ]
        # Gradients are not faster in parallel. So this is running with a pool
        grad_log_lik = self._run_processes(grad_worker_list, shared_obj=theta)

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
        if self.pool is not None:
            batched_idxs = get_batched_list(range(len(grad_log_lik)), self.pool._processes * 2)
        else:
            batched_idxs = [range(len(grad_log_lik))]
        score_score_worker_list = [
            ScoreScoreWorker([grad_log_lik[j] for j in idxs])
            for idxs in batched_idxs
        ]
        tot_score_score = _get_parallel_sum(score_score_worker_list)
        log.info("Obtained score scores %s" % (time.time() - st))

        # Calculate the cross scores (third summand)
        # Instead of calculating \sum_{i \neq j} ES_i ES_j^T directly we calculate
        # (\sum_{i} ES_i) ( \sum_{i} ES_i)^T - \sum_{i} ES_i ES_i^T
        if self.pool is not None:
            batched_labels = get_batched_list(sorted_sample_labels, self.pool._processes * 2)
        else:
            batched_labels = [sorted_sample_labels]
        expected_score_worker_list = [
                ExpectedScoreScoreWorker([expected_scores[l] for l in labels])
                for labels in batched_labels]
        tot_expected_score_score = _get_parallel_sum(expected_score_worker_list)
        tot_cross_expected_scores = expected_scores_sum * expected_scores_sum.T - tot_expected_score_score
        log.info("Obtained cross scores %s" % (time.time() - st))

        # Calculate the hessian (first summand)
        assert(len(self.precalc_data) == len(grad_log_lik))
        hessian_worker_list = [
            HessianWorker(
                [self.precalc_data[j] for j in idxs],
                self.per_target_model)
            for idxs in batched_idxs
        ]
        hessian_sum = _get_parallel_sum(hessian_worker_list, theta)
        log.info("Obtained Hessian %s" % (time.time() - st))

        fisher_info = 1.0/self.num_reps_per_obs * (- hessian_sum - tot_score_score) - np.power(self.num_reps_per_obs, -2.0) * tot_cross_expected_scores
        return fisher_info, -1.0/self.num_samples * hessian_sum

    def _get_gradient_log_lik(self, theta):
        """
        NOTE: parallel is not faster if not a lot of data

        @param theta: the theta to calculate the likelihood for

        Calculate the gradient of the negative log likelihood
        """
        if self.pool is not None:
            batched_data = get_batched_list(self.precalc_data, self.pool._processes * 2)
        else:
            batched_data = [self.precalc_data]
        worker_list = [
            GradientWorker(batch, self.per_target_model)
            for batch in batched_data
        ]
        grad_ll_raw = self._run_processes(
                worker_list,
                theta,
                pool=self.pool if len(worker_list) > 10000 else None)

        grad_ll_raw = np.array(grad_ll_raw)
        grad_ll_dtheta = np.sum(grad_ll_raw, axis=0)

        # Zero out all gradients that affect the constant theta values.
        if self.zero_theta_mask is not None:
            grad_ll_dtheta[self.zero_theta_mask] = 0

        return -1.0/self.num_samples * grad_ll_dtheta
