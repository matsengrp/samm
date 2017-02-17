import time
from multiprocessing import Pool
import numpy as np
import scipy as sp
import logging as log
from survival_problem import SurvivalProblem
from parallel_worker import *
from common import *
from profile_support import profile

class SurvivalProblemCustom(SurvivalProblem):
    """
    Our own implementation to solve the survival problem
    """
    print_iter = 10 # print status every `print_iter` iterations

    # TODO: Think about grouping samples together if they are from the same gibbs sampler
    def __init__(self, feat_generator, samples, penalty_param, theta_mask, num_threads=1):
        assert(penalty_param >= 0)
        self.feature_generator = feat_generator
        self.motif_len = feat_generator.motif_len
        self.samples = samples
        self.theta_mask = theta_mask
        self.per_target_model = self.theta_mask.shape[1] == NUM_NUCLEOTIDES
        self.num_samples = len(self.samples)
        self.feature_mut_steps_pair = [
            (
                sample,
                # generate features, ignore the second return value from create_for_mutation_steps
                self.feature_generator.create_for_mutation_steps(sample)[0],
            )
            for sample in samples
        ]
        self.penalty_param = penalty_param
        self.pool = None
        self.num_threads = num_threads

        self.post_init()

    def post_init(self):
        return

    def get_value(self, theta):
        """
        @return negative penalized log likelihood
        """
        raise NotImplementedError()

    def get_log_lik(self, theta):
        """
        @return negative penalized log likelihood
        """
        log_lik = np.sum([
            SurvivalProblemCustom.calculate_per_sample_log_lik(theta, sample, feature_mut_steps, self.motif_len)
            for sample, feature_mut_steps in self.feature_mut_steps_pair
        ])
        return 1.0/self.num_samples * log_lik

    def calculate_log_lik_ratio_vec(self, theta, prev_theta):
        llr_vec = np.zeros(self.num_samples)
        for sample_id, (sample, feature_mut_steps) in enumerate(self.feature_mut_steps_pair):
            llr_vec[sample_id] = SurvivalProblemCustom.calculate_per_sample_log_lik(theta, sample, feature_mut_steps, self.motif_len) - \
                    SurvivalProblemCustom.calculate_per_sample_log_lik(prev_theta, sample, feature_mut_steps, self.motif_len)
        return llr_vec

    def _get_log_lik_parallel(self, theta):
        """
        @return negative penalized log likelihood
        """
        if self.pool is None:
            raise ValueError("Pool has not been initialized")

        rand_seed = get_randint()
        worker_list = [
            ObjectiveValueWorker(rand_seed + i, theta, sample, feature_vecs, self.motif_len)
            for i, (sample, feature_vecs) in enumerate(self.feature_mut_steps_pair)
        ]
        if self.num_threads > 1:
            ll = self.pool.map(run_multiprocessing_worker, worker_list)
        else:
            ll = map(run_multiprocessing_worker, worker_list)
        return 1.0/self.num_samples * np.sum(ll)

    def _get_gradient_log_lik(self, theta):
        """
        @param theta: where to take the gradient of the total log likelihood

        @return the gradient of the total log likelihood wrt theta
        """
        if self.pool is None:
            raise ValueError("Pool has not been initialized")

        rand_seed = get_randint()
        worker_list = [
            GradientWorker(rand_seed + i, theta, sample, feature_vecs, self.motif_len)
            for i, (sample, feature_vecs) in enumerate(self.feature_mut_steps_pair)
        ]
        if self.num_threads > 1:
            l = self.pool.map(run_multiprocessing_worker, worker_list)
        else:
            l = map(run_multiprocessing_worker, worker_list)
        grad_ll_dtheta = np.sum(l, axis=0)
        return -1.0/self.num_samples * grad_ll_dtheta

    @staticmethod
    def calculate_per_sample_log_lik(theta, sample, feature_mutation_steps, motif_len):
        """
        Calculate the log likelihood of this sample - tries to minimize recalculations as much as possible
        """
        log_lik = 0
        max_pos = sample.obs_seq_mutation.seq_len - 1

        # Store the exp terms so we don't need to recalculate things
        # Gets updated when a position changes
        exp_terms = np.exp(feature_mutation_steps.get_init_theta_sum(theta))
        exp_sum = exp_terms.sum()

        prev_pos_changed = set() # track positions that have mutated already
        prev_mutating_pos = None
        for mut_step, mutating_pos in enumerate(sample.mutation_order):
            if prev_mutating_pos is not None:
                # Find the positions near the position that just mutated
                # TODO: Assumes that the motif lengths are odd!
                change_pos_min = max(prev_mutating_pos - motif_len/2, 0)
                change_pos_max = min(prev_mutating_pos + motif_len/2, max_pos)
                change_pos = np.arange(change_pos_min, change_pos_max + 1)

                # Remove old exp terms from the exp sum
                exp_sum -= exp_terms[change_pos,].sum()
                # Update the exp terms for the positions near the position that just mutated
                for p in change_pos:
                    if p == prev_mutating_pos:
                        # if it is previously mutated position, should be removed from risk group
                        # therefore exp term should be set to zero
                        exp_terms[prev_mutating_pos,] = 0
                    elif p not in prev_pos_changed:
                        # position hasn't mutated yet.
                        # need to update its exp value
                        exp_terms[p,] = np.exp(feature_mutation_steps.get_theta_sum(mut_step, p, theta))
                # Add in the new exp terms from the exp sum
                exp_sum += exp_terms[change_pos].sum()

            # Determine the theta column
            col_idx = 0
            if theta.shape[1] == NUM_NUCLEOTIDES:
                # Get the column for this target nucleotide
                col_idx = NUCLEOTIDE_DICT[sample.obs_seq_mutation.end_seq[mutating_pos]]

            # Add to the log likelihood the log likelihood at this step: theta * psi - log(sum(exp terms))
            log_lik += feature_mutation_steps.get_theta_sum(mut_step, mutating_pos, theta, col_idx) - np.log(exp_sum)

            prev_mutating_pos = mutating_pos
            prev_pos_changed.add(mutating_pos)
        return log_lik

    @staticmethod
    def get_gradient_log_lik_per_sample(theta, sample, feature_mutation_steps, motif_len):
        """
        @param feature_mutation_steps: FeatureMutationSteps
        """
        if self.pool is None:
            raise ValueError("Pool has not been initialized")

        rand_seed = get_randint()
        worker_list = [
            GradientWorker(rand_seed + i, theta, sample, feature_vecs, self.motif_len)
            for i, (sample, feature_vecs) in enumerate(self.feature_vec_sample_pair)
        ]
        if self.num_threads > 1:
            l = self.pool.map(run_multiprocessing_worker, worker_list)
        else:
            l = map(run_multiprocessing_worker, worker_list)

        grad_ll_dtheta = np.sum(l, axis=0)
        return -1.0/self.num_samples * grad_ll_dtheta

    @staticmethod
    def get_gradient_log_lik_per_sample(theta, sample, feature_mutation_steps, motif_len):
        grad = np.zeros(theta.shape)
        max_pos = sample.obs_seq_mutation.seq_len - 1

        # Store the exp terms so we don't need to recalculate things
        # Gets updated when a position changes
        exp_terms = np.exp(feature_mutation_steps.get_init_theta_sum(theta))
        exp_sum = exp_terms.sum()

        # Store the gradient vector (not normalized) so we don't need to recalculate things
        grad_log_sum_exp = feature_mutation_steps.feat_matrix0T * exp_terms

        prev_pos_changed = set() # track positions that have mutated already
        prev_mutating_pos = None
        prev_vecs_at_mutation_step = None
        for mutating_pos, vecs_at_mutation_step in zip(sample.mutation_order, feature_mutation_steps.feature_vec_dicts):
            if prev_mutating_pos is not None:
                # Find the positions near the position that just mutated
                # TODO: Assumes that the motif lengths are odd!
                change_pos_min = max(prev_mutating_pos - motif_len/2, 0)
                change_pos_max = min(prev_mutating_pos + motif_len/2, max_pos)
                change_pos = np.arange(change_pos_min, change_pos_max + 1)

                # Remove old exp terms from the exp sum
                exp_sum -= exp_terms[change_pos,].sum()
                # Update the exp terms for the positions near the position that just mutated
                for p in change_pos:
                    if p == prev_mutating_pos:
                        # if it is previously mutated position, should be removed from risk group
                        # therefore exp term should be set to zero
                        grad_log_sum_exp[prev_vecs_at_mutation_step[p],] -= exp_terms[p,]
                        exp_terms[p,] = 0
                    elif p not in prev_pos_changed:
                        # position hasn't mutated yet.
                        # need to update its exp value and the corresponding gradient vector
                        grad_log_sum_exp[prev_vecs_at_mutation_step[p],] -= exp_terms[p,]
                        curr_feat_vec = vecs_at_mutation_step[p]
                        exp_terms[p,] = np.exp(theta[curr_feat_vec,])
                        grad_log_sum_exp[curr_feat_vec,] += exp_terms[p,]
                        # add new exp term to gradient and remove old exp term from gradient
                # Add in the new exp terms from the exp sum
                exp_sum += exp_terms[change_pos,].sum()

            # Add to the gradient the gradient at this step
            col_idx = 0
            if theta.shape[1] == NUM_NUCLEOTIDES:
                col_idx = NUCLEOTIDE_DICT[sample.obs_seq_mutation.end_seq[mutating_pos]]

            grad[vecs_at_mutation_step[mutating_pos], col_idx] += 1
            grad -= grad_log_sum_exp/exp_sum

            prev_mutating_pos = mutating_pos
            prev_vecs_at_mutation_step = vecs_at_mutation_step
            prev_pos_changed.add(mutating_pos)
        return grad

class GradientWorker(ParallelWorker):
    """
    Stores the information for calculating gradient
    """
    def __init__(self, seed, theta, sample, feature_vecs, motif_len):
        """
        @param theta: where to take the gradient of the total log likelihood
        @param sample: class ImputedSequenceMutations
        @param feature_vecs: list of sparse feature vectors for all at-risk positions at every mutation step
        """
        self.seed = seed
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
    def __init__(self, seed, theta, sample, feature_vecs, motif_len):
        """
        @param theta: where to take the gradient of the total log likelihood
        @param sample: class ImputedSequenceMutations
        @param feature_vecs: list of sparse feature vectors for all at-risk positions at every mutation step
        """
        self.seed = seed
        self.theta = theta
        self.sample = sample
        self.feature_vecs = feature_vecs
        self.motif_len = motif_len

    def run(self):
        """
        @return the log likelihood for this sample
        """
        return SurvivalProblemCustom.calculate_per_sample_log_lik(self.theta, self.sample, self.feature_vecs, self.motif_len)
