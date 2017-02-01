import time
from multiprocessing import Pool
import numpy as np
import scipy as sp
import logging as log
from survival_problem import SurvivalProblem
from common import soft_threshold
from parallel_worker import *
from common import measure_time

class SurvivalProblemCustom(SurvivalProblem):
    """
    Our own implementation to solve the survival problem
    """
    print_iter = 10 # print status every `print_iter` iterations

    def __init__(self, feat_generator, samples, penalty_param):
        """
        @param feat_generator: feature generator
        @param init_theta: where to initialize the gradient descent procedure from
        @param penalty_param: the lasso parameter. should be non-negative
        """
        assert(penalty_param >= 0)

        self.feature_generator = feat_generator
        self.motif_len = feat_generator.motif_len
        self.samples = samples
        self.num_samples = len(self.samples)
        self.feature_vec_sample_pair = [
            (
                sample,
                self.feature_generator.create_for_mutation_steps(sample)[0],
            )
            for sample in samples
        ]
        self.penalty_param = penalty_param

        self.post_init()

    def post_init(self):
        return

    def get_log_lik(self, theta):
        """
        @return negative penalized log likelihood
        """
        log_lik = np.sum([
            SurvivalProblemCustom.calculate_per_sample_log_lik(theta, sample, feature_vecs, self.motif_len)
            for sample, feature_vecs in self.feature_vec_sample_pair
        ])
        return 1.0/self.num_samples * log_lik

    def calculate_log_lik_ratio_vec(self, theta, prev_theta):
        llr_vec = np.zeros(self.num_samples)
        for sample_id, (sample, feature_vecs) in enumerate(self.feature_vec_sample_pair):
            llr_vec[sample_id] = SurvivalProblemCustom.calculate_per_sample_log_lik(theta, sample, feature_vecs, self.motif_len) - \
                    SurvivalProblemCustom.calculate_per_sample_log_lik(prev_theta, sample, feature_vecs, self.motif_len)
        return llr_vec

    @staticmethod
    def calculate_per_sample_log_lik(theta, sample, feature_vecs, motif_len):
        obj = 0
        denom_terms = np.exp([theta[one_feats].sum() for one_feats in feature_vecs[0].values()])
        denom_sum = denom_terms.sum()

        num_pos = len(feature_vecs[0]) - 1

        prev_mutating_pos = None
        prev_vecs_at_mutation_step = None
        for mutating_pos, vecs_at_mutation_step in zip(sample.mutation_order, feature_vecs):
            if prev_mutating_pos is not None:
                change_pos_min = max(prev_mutating_pos - motif_len/2, 0)
                change_pos_max = min(prev_mutating_pos + motif_len/2, num_pos)
                change_pos = np.arange(change_pos_min, change_pos_max)

                denom_sum -= denom_terms[change_pos].sum()
                # change position values
                for p in change_pos:
                    if denom_terms[p] != 0:
                        denom_terms[p] = np.exp(theta[vecs_at_mutation_step[p]].sum())

                denom_sum += denom_terms[change_pos].sum()

            obj += theta[vecs_at_mutation_step[mutating_pos]].sum()
            obj -= np.log(denom_sum)

            denom_sum -= denom_terms[mutating_pos]
            denom_terms[mutating_pos] = 0

            prev_mutating_pos = mutating_pos
            prev_vecs_at_mutation_step = vecs_at_mutation_step
        return obj

    def _get_log_lik_parallel(self, theta):
        """
        @return negative penalized log likelihood
        """
        ll = self.pool.map(
            run_parallel_worker,
            [ObjectiveValueWorker(theta, sample, feature_vecs, self.motif_len) for sample, feature_vecs in self.feature_vec_sample_pair]
        )
        return 1.0/self.num_samples * np.sum(ll)

    def _get_gradient_log_lik(self, theta):
        """
        @param theta: where to take the gradient of the total log likelihood

        @return the gradient of the total log likelihood wrt theta
        """
        l = self.pool.map(
            run_parallel_worker,
            [
                GradientWorker(theta, sample, feature_vecs, self.motif_len)
                for sample, feature_vecs in self.feature_vec_sample_pair
            ]
        )
        grad_ll_dtheta = np.sum(l, axis=0)
        return -1.0/self.num_samples * grad_ll_dtheta

    @staticmethod
    def get_gradient_log_lik_per_sample(theta, sample, feature_vecs, motif_len):
        grad = np.zeros(theta.size)
        num_pos = len(feature_vecs[0]) - 1

        denom_terms = np.exp([theta[one_feats].sum() for one_feats in feature_vecs[0].values()])
        denom_sum = denom_terms.sum()

        grad_log_sum_exp = np.zeros(theta.size)
        for pos, feat_vec in feature_vecs[0].iteritems():
            grad_log_sum_exp[feat_vec] += denom_terms[pos]

        prev_mutating_pos = None
        prev_vecs_at_mutation_step = None
        for mutating_pos, vecs_at_mutation_step in zip(sample.mutation_order, feature_vecs):
            if prev_mutating_pos is not None:
                change_pos_min = max(prev_mutating_pos - motif_len/2, 0)
                change_pos_max = min(prev_mutating_pos + motif_len/2, num_pos)
                change_pos = np.arange(change_pos_min, change_pos_max)

                denom_sum -= denom_terms[change_pos].sum()
                # change position values
                for p in change_pos:
                    if denom_terms[p] != 0:
                        grad_log_sum_exp[prev_vecs_at_mutation_step[p]] -= denom_terms[p]
                        denom_terms[p] = np.exp(theta[vecs_at_mutation_step[p]].sum())
                        grad_log_sum_exp[vecs_at_mutation_step[p]] += denom_terms[p]

                denom_sum += denom_terms[change_pos].sum()

            grad[vecs_at_mutation_step[mutating_pos]] += 1

            grad -= grad_log_sum_exp/denom_sum

            # change current position
            denom_sum -= denom_terms[mutating_pos]
            grad_log_sum_exp[vecs_at_mutation_step[mutating_pos]] -= denom_terms[mutating_pos]
            denom_terms[mutating_pos] = 0

            prev_mutating_pos = mutating_pos
            prev_vecs_at_mutation_step = vecs_at_mutation_step
        return grad

class GradientWorker(ParallelWorker):
    """
    Stores the information for calculating gradient
    """
    def __init__(self, theta, sample, feature_vecs, motif_len):
        """
        @param theta: where to take the gradient of the total log likelihood
        @param sample: class ImputedSequenceMutations
        @param feature_vecs: list of sparse feature vectors for all at-risk positions at every mutation step
        """
        self.theta = theta
        self.sample = sample
        self.feature_vecs = feature_vecs
        self.motif_len = motif_len

    def run(self):
        """
        @return the gradient of the log likelihood for this sample
        """
        return SurvivalProblemCustom.get_gradient_log_lik_per_sample(self.theta, self.sample, self.feature_vecs, self.motif_len)

class ObjectiveValueWorker(ParallelWorker):
    """
    Stores the information for calculating objective function value
    """
    def __init__(self, theta, sample, feature_vecs, motif_len):
        """
        @param theta: where to take the gradient of the total log likelihood
        @param sample: class ImputedSequenceMutations
        @param feature_vecs: list of sparse feature vectors for all at-risk positions at every mutation step
        """
        self.theta = theta
        self.sample = sample
        self.feature_vecs = feature_vecs
        self.motif_len = motif_len

    def run(self):
        """
        @return the log likelihood for this sample
        """
        return SurvivalProblemCustom.calculate_per_sample_log_lik(self.theta, self.sample, self.feature_vecs, self.motif_len)
