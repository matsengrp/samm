import time
from multiprocessing import Pool
import numpy as np
import scipy as sp
import logging as log
from survival_problem import SurvivalProblem
from common import soft_threshold
from parallel_worker import *

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
            SurvivalProblemCustom.calculate_per_sample_log_lik(theta, sample, feature_vecs)
            for sample, feature_vecs in self.feature_vec_sample_pair
        ])
        return 1.0/self.num_samples * log_lik

    def calculate_log_lik_ratio_vec(self, theta, prev_theta):
        llr_vec = np.zeros(self.num_samples)
        for sample_id, (sample, feature_vecs) in enumerate(self.feature_vec_sample_pair):
            llr_vec[sample_id] = SurvivalProblemCustom.calculate_per_sample_log_lik(theta, sample, feature_vecs) - \
                    SurvivalProblemCustom.calculate_per_sample_log_lik(prev_theta, sample, feature_vecs)
        return llr_vec

    @staticmethod
    def calculate_per_sample_log_lik(theta, sample, feature_vecs):
        """
        @param sample: instance of class ImputedSequenceMutations
        @param feature_vecs: list of sparse feature vectors for all at-risk positions at every mutation step
        @return the log likelihood of theta for the given sample
        """
        obj = 0
        for mutating_pos, vecs_at_mutation_step in zip(sample.mutation_order, feature_vecs):
            # vecs_at_mutation_step[i] are the feature vectors of the at-risk group after mutation i
            feature_vec_mutated = vecs_at_mutation_step[mutating_pos]
            obj += np.sum(theta[feature_vec_mutated]) - sp.misc.logsumexp(
                [np.sum(theta[f]) for f in vecs_at_mutation_step.values()]
            )
        return obj

    def _get_log_lik_parallel(self, theta):
        """
        @return negative penalized log likelihood
        """
        ll = self.pool.map(
            run_parallel_worker,
            [ObjectiveValueWorker(theta, sample, feature_vecs) for sample, feature_vecs in self.feature_vec_sample_pair]
        )
        return 1.0/self.num_samples * np.sum(ll)

    def _get_gradient_log_lik(self, theta):
        """
        @param theta: where to take the gradient of the total log likelihood

        @return the gradient of the total log likelihood wrt theta
        """
        l = self.pool.map(
            run_parallel_worker,
            [GradientWorker(theta, sample, feature_vecs) for sample, feature_vecs in self.feature_vec_sample_pair]
        )
        grad_ll_dtheta = np.sum(l, axis=0)
        return -1.0/self.num_samples * grad_ll_dtheta

    @staticmethod
    def get_gradient_log_lik_per_sample(theta, sample, feature_vecs):
        grad = np.zeros(theta.size)
        for mutating_pos, vecs_at_mutation_step in zip(sample.mutation_order, feature_vecs):
            grad[vecs_at_mutation_step[mutating_pos]] += 1
            denom = 0
            grad_log_sum_exp = np.zeros(theta.size)
            denom = np.exp([theta[one_feats].sum() for one_feats in vecs_at_mutation_step.values()]).sum()
            for one_feats in vecs_at_mutation_step.values():
                val = np.exp(theta[one_feats].sum())
                for f in one_feats:
                    grad_log_sum_exp[f] += val
            grad -= grad_log_sum_exp/denom
        return grad

class GradientWorker(ParallelWorker):
    """
    Stores the information for calculating gradient
    """
    def __init__(self, theta, sample, feature_vecs):
        """
        @param theta: where to take the gradient of the total log likelihood
        @param sample: class ImputedSequenceMutations
        @param feature_vecs: list of sparse feature vectors for all at-risk positions at every mutation step
        """
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
    def __init__(self, theta, sample, feature_vecs):
        """
        @param theta: where to take the gradient of the total log likelihood
        @param sample: class ImputedSequenceMutations
        @param feature_vecs: list of sparse feature vectors for all at-risk positions at every mutation step
        """
        self.theta = theta
        self.sample = sample
        self.feature_vecs = feature_vecs

    def run(self):
        """
        @return the log likelihood for this sample
        """
        return SurvivalProblemCustom.calculate_per_sample_log_lik(self.theta, self.sample, self.feature_vecs)
