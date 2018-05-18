import threading
import time
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix, dok_matrix

import logging as log
from survival_problem import SurvivalProblem
from survival_problem_grad_descent_workers import *
from common import *
from profile_support import profile

class SurvivalProblemCustom(SurvivalProblem):
    """
    Our own implementation to solve the survival problem
    """
    print_iter = 10 # print status every `print_iter` iterations

    def __init__(self, feat_generator, samples, sample_labels=None, penalty_params=[0], per_target_model=False, possible_theta_mask=None, zero_theta_mask=None, fuse_windows=[], fuse_center_only=False, max_threads=1):
        """
        @param sample_labels: only used for calculating the Hessian
        @param possible_theta_mask: these theta values are some finite number
        @param zero_theta_mask: these theta values are forced to be zero
        @param max_threads: this specifies number of threads to create via the python threading library
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
        self.max_threads = max_threads

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
        precalc_data = [None for _ in samples]
        worker_list = [
            PrecalcDataWorker(
                precalc_data,
                i,
                sample,
                self.feature_generator.create_for_mutation_steps(sample),
                self.feature_generator.feature_vec_len,
                self.per_target_model,
            ) for i, sample in enumerate(samples)
        ]
        self._run_threads(worker_list)
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

    def _run_threads(self, worker_list):
        """
        Run and join python Thread objects

        @param worker_list: a list of Worker objects from down there
        """
        # Batching so not to kill all mymemory
        batched_worker_list = [
                ThreadWorker(l) for l in get_batched_list(worker_list, self.max_threads)]
        for w in batched_worker_list:
            w.start()
        for w in batched_worker_list:
            w.join()

    def _get_log_lik_parallel(self, theta):
        """
        JUST KIDDING - parallel is not faster
        @param theta: the theta to calculate the likelihood for
        @return vector of log likelihood values
        """
        worker_results = [None for _ in self.precalc_data]
        worker_list = [
            ObjectiveValueWorker(
                worker_results,
                i,
                sample_data,
                self.per_target_model,
                theta) for i, sample_data in enumerate(self.precalc_data)
        ]
        self._run_threads(worker_list)

        return np.array(worker_results)

    def get_hessian(self, theta):
        """
        Uses Louis's method to calculate the information matrix of the observed data

        @return fishers information matrix of the observed data, hessian of the log likelihood of the complete data
        """
        def _get_parallel_sum(worker_list, res_list):
            self._run_threads(worker_list)
            assert not any([r is None for r in res_list])
            tot = 0
            for r in res_list:
                tot += r
            return tot

        st = time.time()
        grad_log_lik = [None for _ in self.precalc_data]
        grad_worker_list = [
            GradientWorker(
                grad_log_lik,
                i,
                sample_data,
                self.per_target_model,
                theta) for i, sample_data in enumerate(self.precalc_data)
        ]
        self._run_threads(grad_worker_list)

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
        num_batches = int(len(grad_log_lik)/self.max_threads) * 2
        batched_idxs = get_batched_list(range(len(grad_log_lik)), num_batches)
        score_scores = [None for _ in batched_idxs]
        score_score_worker_list = [
            ScoreScoreWorker(
                score_scores,
                i,
                [grad_log_lik[j] for j in idxs])
            for i, idxs in enumerate(batched_idxs)
        ]
        tot_score_score = _get_parallel_sum(score_score_worker_list, score_scores)
        log.info("Obtained score scores %s" % (time.time() - st))

        # Calculate the cross scores (third summand)
        # Instead of calculating \sum_{i \neq j} ES_i ES_j^T directly we calculate
        # (\sum_{i} ES_i) ( \sum_{i} ES_i)^T - \sum_{i} ES_i ES_i^T
        batched_labels = get_batched_list(sorted_sample_labels, num_batches)
        exp_score_scores = [None for _ in batched_labels]
        expected_score_worker_list = [
                ExpectedScoreScoreWorker(
                    exp_score_scores,
                    i,
                    labels,
                    expected_scores)
                for i, labels in enumerate(batched_labels)]
        tot_expected_score_score = _get_parallel_sum(
                expected_score_worker_list,
                exp_score_scores)
        tot_cross_expected_scores = expected_scores_sum * expected_scores_sum.T - tot_expected_score_score
        log.info("Obtained cross scores %s" % (time.time() - st))

        # Calculate the hessian (first summand)
        assert(len(self.precalc_data) == len(grad_log_lik))
        hessian_res = [None for _ in batched_idxs]
        hessian_worker_list = [
            HessianWorker(
                hessian_res,
                i,
                [self.precalc_data[j] for j in idxs],
                self.per_target_model,
                theta)
            for i, idxs in enumerate(batched_idxs)
        ]
        hessian_sum = _get_parallel_sum(hessian_worker_list, hessian_res)
        log.info("Obtained Hessian %s" % (time.time() - st))

        fisher_info = 1.0/self.num_reps_per_obs * (- hessian_sum - tot_score_score) - np.power(self.num_reps_per_obs, -2.0) * tot_cross_expected_scores
        return fisher_info, -1.0/self.num_samples * hessian_sum

    def _get_gradient_log_lik(self, theta):
        """
        JUST KIDDING - parallel is not faster
        @param theta: the theta to calculate the likelihood for

        Calculate the gradient of the negative log likelihood - delegates to separate cpu threads if threads > 1
        """
        rand_seed = get_randint()
        grad_ll_raw = [None for _ in self.precalc_data]
        worker_list = [
            GradientWorker(
                grad_ll_raw,
                i,
                sample_data,
                self.per_target_model,
                theta)
            for i, sample_data in enumerate(self.precalc_data)
        ]
        self._run_threads(worker_list)

        grad_ll_raw = np.array(grad_ll_raw)
        grad_ll_dtheta = np.sum(grad_ll_raw, axis=0)

        # Zero out all gradients that affect the constant theta values.
        if self.zero_theta_mask is not None:
            grad_ll_dtheta[self.zero_theta_mask] = 0

        return -1.0/self.num_samples * grad_ll_dtheta
