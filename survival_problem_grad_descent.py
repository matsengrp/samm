import time
import traceback
from multiprocessing import Pool
import numpy as np
import scipy as sp
import logging as log
from survival_problem import SurvivalProblem
from common import soft_threshold

class SurvivalProblemGradientDescent(SurvivalProblem):
    """
    Our own implementation of proximal gradient descent to solve the survival problem
    Objective function: - log likelihood of theta + lasso penalty on theta
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

    def get_value(self, theta):
        """
        @return negative penalized log likelihood
        """
        log_lik = np.sum([
            SurvivalProblemGradientDescent.calculate_per_sample_log_lik(theta, sample, feature_vecs)
            for sample, feature_vecs in self.feature_vec_sample_pair
        ])
        return -(1.0/self.num_samples * log_lik - self.penalty_param * np.linalg.norm(theta, ord=1))

    def calculate_log_lik_ratio_vec(self, theta, prev_theta):
        llr_vec = np.zeros(self.num_samples)
        for sample_id, (sample, feature_vecs) in enumerate(self.feature_vec_sample_pair):
            llr_vec[sample_id] = SurvivalProblemGradientDescent.calculate_per_sample_log_lik(theta, sample, feature_vecs) - \
                    SurvivalProblemGradientDescent.calculate_per_sample_log_lik(prev_theta, sample, feature_vecs)
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

    def solve(self, init_theta, max_iters=1000, num_threads=1, init_step_size=1, step_size_shrink=0.5, backtrack_alpha = 0.01, diff_thres=1e-6, verbose=False):
        """
        Runs proximal gradient descent to minimize the negative penalized log likelihood

        @param init_theta: where to initialize the gradient descent
        @param max_iters: maximum number of iterations of gradient descent
        @param init_step_size: how big to initialize the step size factor
        @param step_size_shrink: how much to shrink the step size during backtracking line descent
        @param backtrack_alpha: the alpha in backtracking line descent (p464 in Boyd)
        @param diff_thres: if the difference is less than diff_thres, then stop gradient descent
        @param verbose: whether to print out the status at each iteration
        @return final fitted value of theta and penalized log likelihood
        """
        st = time.time()
        self.pool = Pool(num_threads)
        theta = init_theta
        step_size = init_step_size
        current_value = self.get_value(theta)
        for i in range(max_iters):
            if i % self.print_iter == 0:
                log.info("GD iter %d, val %f, time %d" % (i, current_value, time.time() - st))
            # Calculate gradient of the smooth part
            grad = self._get_gradient_log_lik(theta)
            potential_theta = theta - step_size * grad
            # Do proximal gradient step
            potential_theta = soft_threshold(potential_theta, step_size * self.penalty_param)
            potential_value = self.get_value_parallel(potential_theta)

            # Do backtracking line search
            expected_decrease = backtrack_alpha * np.power(np.linalg.norm(grad), 2)
            while potential_value >= current_value - step_size * expected_decrease:
                if step_size * expected_decrease < diff_thres:
                    # Stop if difference in objective function is too small
                    break
                step_size *= step_size_shrink
                potential_theta = theta - step_size * grad
                # Do proximal gradient step
                potential_theta = soft_threshold(potential_theta, step_size * self.penalty_param)
                potential_value = self.get_value_parallel(potential_theta)

            if potential_value > current_value:
                # Stop if value is increasing
                break
            else:
                theta = potential_theta
                diff = current_value - potential_value
                current_value = potential_value
                if diff < diff_thres:
                    # Stop if difference in objective function is too small
                    break
        self.pool.close()
        log.info("final GD iter %d, val %f, time %d" % (i, current_value, time.time() - st))
        return theta, -current_value

    def get_value_parallel(self, theta):
        """
        @return negative penalized log likelihood
        """
        ll = self.pool.map(
            _run_value_worker,
            [Worker(theta, sample, feature_vecs) for sample, feature_vecs in self.feature_vec_sample_pair]
        )
        return -(1.0/self.num_samples * np.sum(ll) - self.penalty_param * np.linalg.norm(theta, ord=1))


    def _get_gradient_log_lik(self, theta):
        """
        @param theta: where to take the gradient of the total log likelihood

        @return the gradient of the total log likelihood wrt theta
        """
        l = self.pool.map(
            _run_worker,
            [Worker(theta, sample, feature_vecs) for sample, feature_vecs in self.feature_vec_sample_pair]
        )
        grad_ll_dtheta = np.sum(l, axis=0)
        return -1.0/self.num_samples * grad_ll_dtheta

def _run_worker(worker):
    """
    @param worker: Worker
    Function called by each worker process in the multiprocessing pool
    Note: this must be a global function
    """
    result = None
    try:
        result = worker.run()
    except Exception as e:
        print "Exception caught: %s" % e
        traceback.print_exc()
    return result

def _run_value_worker(worker):
    """
    @param worker: Worker
    Function called by each worker process in the multiprocessing pool
    Note: this must be a global function
    """
    result = None
    try:
        result = worker.run_value()
    except Exception as e:
        print "Exception caught: %s" % e
        traceback.print_exc()
    return result


class Worker:
    """
    Stores the information for running a sampler
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
        theta = self.theta
        sample = self.sample
        feature_vecs = self.feature_vecs

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

    def run_value(self):
        return SurvivalProblemGradientDescent.calculate_per_sample_log_lik(
            self.theta, self.sample, self.feature_vecs)
