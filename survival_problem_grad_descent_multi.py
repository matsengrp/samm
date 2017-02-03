import time
from multiprocessing import Pool
import numpy as np
import scipy as sp
import logging as log
from survival_problem import SurvivalProblem
from common import *
from parallel_worker import *

class SurvivalProblemCustomMulti(SurvivalProblem):
    """
    Our own implementation to solve the survival problem
    """
    print_iter = 10 # print status every `print_iter` iterations

    def __init__(self, feat_generator, samples, penalty_param, theta_mask):
        """
        @param feat_generator: feature generator
        @param init_theta: where to initialize the gradient descent procedure from
        @param penalty_param: the lasso parameter. should be non-negative
        """
        assert(penalty_param >= 0)

        self.feature_generator = feat_generator
        self.motif_len = feat_generator.motif_len
        motif_list = feat_generator.get_motif_list()
        self.theta_mask = theta_mask

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
            SurvivalProblemCustomMulti.calculate_per_sample_log_lik(theta, sample, feature_vecs, self.motif_len)
            for sample, feature_vecs in self.feature_vec_sample_pair
        ])
        return 1.0/self.num_samples * log_lik

    def calculate_log_lik_ratio_vec(self, theta, prev_theta):
        llr_vec = np.zeros(self.num_samples)
        for sample_id, (sample, feature_vecs) in enumerate(self.feature_vec_sample_pair):
            llr_vec[sample_id] = SurvivalProblemCustomMulti.calculate_per_sample_log_lik(theta, sample, feature_vecs, self.motif_len) - \
                    SurvivalProblemCustomMulti.calculate_per_sample_log_lik(prev_theta, sample, feature_vecs, self.motif_len)
        return llr_vec

    @staticmethod
    def calculate_per_sample_log_lik(theta, sample, feature_vecs, motif_len):
        """
        Calculate the log likelihood of this sample - tries to minimize recalculations as much as possible
        """
        log_lik = 0
        max_pos = len(feature_vecs[0]) - 1

        # Store the exp terms so we don't need to recalculate things
        # Gets updated when a position changes
        exp_terms = np.exp(
            [
                [theta[one_feats, i].sum() for i in range(NUM_NUCLEOTIDES)]
                for one_feats in feature_vecs[0].values()
            ]
        )
        # The sum of the exp terms. Update the sum accordingly to minimize recalculation
        exp_sum = exp_terms.sum()

        pos_changed = [] # track positions that have mutated already
        prev_mutating_pos = None
        prev_vecs_at_mutation_step = None
        for mutating_pos, vecs_at_mutation_step in zip(sample.mutation_order, feature_vecs):
            if prev_mutating_pos is not None:
                # Find the positions near the position that just mutated
                # TODO: Assumes that the motif lengths are odd!
                change_pos_min = max(prev_mutating_pos - motif_len/2, 0)
                change_pos_max = min(prev_mutating_pos + motif_len/2, max_pos)
                change_pos = np.arange(change_pos_min, change_pos_max + 1)

                # Remove old exp terms from the exp sum
                exp_sum -= exp_terms[change_pos, :].sum()
                # Update the exp terms for the positions near the position that just mutated
                for p in change_pos:
                    if p == prev_mutating_pos:
                        # if it is previously mutated position, should be removed from risk group
                        # therefore exp term should be set to zero
                        exp_terms[prev_mutating_pos, :] = 0
                    elif p not in pos_changed:
                        # position hasn't mutated yet.
                        # need to update its exp value
                        exp_terms[p, :] = np.exp([theta[vecs_at_mutation_step[p], i].sum() for i in range(NUM_NUCLEOTIDES)])
                # Add in the new exp terms from the exp sum
                exp_sum += exp_terms[change_pos].sum()

            # Add to the log likelihood the log likelihood at this step: theta * psi - log(sum(exp terms))
            log_lik += theta[
                vecs_at_mutation_step[mutating_pos], # motif idx
                NUCLEOTIDE_DICT[sample.obs_seq_mutation.end_seq[mutating_pos]] # target nucleotide idx
            ].sum() - np.log(exp_sum)

            prev_mutating_pos = mutating_pos
            prev_vecs_at_mutation_step = vecs_at_mutation_step
            pos_changed.append(mutating_pos)
        return log_lik

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
        grad = np.zeros(theta.shape)
        max_pos = len(feature_vecs[0]) - 1

        # Store the exp terms so we don't need to recalculate things
        # Gets updated when a position changes
        exp_terms = np.exp(
            [
                [theta[one_feats, i].sum() for i in range(NUM_NUCLEOTIDES)]
                for one_feats in feature_vecs[0].values()
            ]
        )
        # The sum of the exp terms. Update the sum accordingly to minimize recalculation
        exp_sum = exp_terms.sum()

        # Store the gradient vector (not normalized) so we don't need to recalculate things
        grad_log_sum_exp = np.zeros(theta.shape)
        for pos, feat_vec in feature_vecs[0].iteritems():
            grad_log_sum_exp[feat_vec, :] += exp_terms[pos, :]

        pos_changed = [] # track positions that have mutated already
        prev_mutating_pos = None
        prev_vecs_at_mutation_step = None
        for mutating_pos, vecs_at_mutation_step in zip(sample.mutation_order, feature_vecs):
            if prev_mutating_pos is not None:
                # Find the positions near the position that just mutated
                # TODO: Assumes that the motif lengths are odd!
                change_pos_min = max(prev_mutating_pos - motif_len/2, 0)
                change_pos_max = min(prev_mutating_pos + motif_len/2, max_pos)
                change_pos = np.arange(change_pos_min, change_pos_max + 1)

                # Remove old exp terms from the exp sum
                exp_sum -= exp_terms[change_pos, :].sum()
                # Update the exp terms for the positions near the position that just mutated
                for p in change_pos:
                    if p == prev_mutating_pos:
                        # if it is previously mutated position, should be removed from risk group
                        # therefore exp term should be set to zero
                        grad_log_sum_exp[prev_vecs_at_mutation_step[p]] -= exp_terms[p]
                        exp_terms[p, :] = 0
                    elif p not in pos_changed:
                        # position hasn't mutated yet.
                        # need to update its exp value and the corresponding gradient vector

                        # remove old exp term from gradient
                        grad_log_sum_exp[prev_vecs_at_mutation_step[p], :] -= exp_terms[p, :]
                        exp_terms[p,:] = np.exp(
                            [theta[vecs_at_mutation_step[p], i].sum() for i in range(NUM_NUCLEOTIDES)]
                        )
                        # add new exp term to gradient
                        grad_log_sum_exp[vecs_at_mutation_step[p], :] += exp_terms[p, :]
                # Add in the new exp terms from the exp sum
                exp_sum += exp_terms[change_pos, :].sum()

            # Add the gradient at this step
            grad[vecs_at_mutation_step[mutating_pos], NUCLEOTIDE_DICT[sample.obs_seq_mutation.end_seq[mutating_pos]]] += 1
            grad -= grad_log_sum_exp/exp_sum

            prev_mutating_pos = mutating_pos
            prev_vecs_at_mutation_step = vecs_at_mutation_step
            pos_changed.append(mutating_pos)
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
        return SurvivalProblemCustomMulti.get_gradient_log_lik_per_sample(self.theta, self.sample, self.feature_vecs, self.motif_len)

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
        return SurvivalProblemCustomMulti.calculate_per_sample_log_lik(self.theta, self.sample, self.feature_vecs, self.motif_len)
