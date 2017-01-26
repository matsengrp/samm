import time
import traceback
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
    prox_grad_batch_size = 200

    def __init__(self, feat_generator, samples, lasso_param):
        """
        @param feat_generator: feature generator
        @param init_theta: where to initialize the gradient descent procedure from
        @param lasso_param: the lasso parameter. should be non-negative
        """
        assert(lasso_param >= 0)

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
        self.lasso_param = lasso_param

    def get_value(self, theta):
        """
        @return negative penalized log likelihood
        """
        log_lik = np.sum([
            SurvivalProblemGradientDescent.calculate_per_sample_log_lik(theta, sample, feature_vecs)
            for sample, feature_vecs in self.feature_vec_sample_pair
        ])
        return -(1.0/self.num_samples * log_lik - self.lasso_param * np.linalg.norm(theta, ord=1))

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

    def solve(self, init_theta, max_iters=1000, init_step_size=1, step_size_shrink=0.5, backtrack_alpha = 0.01, prox_diff_thres=0.1, diff_thres=1e-6, verbose=False):
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
        theta = init_theta
        step_size = init_step_size
        current_value = self.get_value(theta)
        finish_gd = False
        do_batch_gd = False
        prox_grad_batch_size = self.prox_grad_batch_size
        for i in range(max_iters):
            if i % self.print_iter == 0:
                log.info("GD iter %d, val %f, time %d" % (i, current_value, time.time() - st))

            # Perform stochastic proximal gradient descent
            permuted_feat_vec_sample_pair = np.random.permutation(self.feature_vec_sample_pair)
            print "prox_grad_batch_size", prox_grad_batch_size
            for start_idx in range(0, self.num_samples, prox_grad_batch_size):
                print "start_idx", start_idx
                print "start_idx + prox_grad_batch_size", start_idx + prox_grad_batch_size
                print "self.num_samples", self.num_samples
                # Calculate gradient of the smooth part
                grad = self._get_gradient_log_lik(theta, permuted_feat_vec_sample_pair[start_idx:start_idx + prox_grad_batch_size])
                potential_theta = theta - step_size * grad
                # Do proximal gradient step
                potential_theta = soft_threshold(potential_theta, step_size * self.lasso_param)
                potential_value = self.get_value(potential_theta)

                # Do backtracking line search
                expected_decrease = backtrack_alpha * np.power(np.linalg.norm(grad), 2)
                while potential_value >= current_value - step_size * expected_decrease:
                    if not do_batch_gd and step_size * expected_decrease < prox_diff_thres:
                        # If proximal gradient descent is no longer decreasing a lot, go to full batch size
                        break
                    elif step_size * expected_decrease < diff_thres:
                        # Stop if difference in objective function is too small
                        break
                    step_size *= step_size_shrink
                    potential_theta = theta - step_size * grad
                    # Do proximal gradient step
                    potential_theta = soft_threshold(potential_theta, step_size * self.lasso_param)
                    potential_value = self.get_value(potential_theta)

                if potential_value > current_value:
                    if do_batch_gd:
                        # Stop if value is increasing
                        finish_gd = True
                    do_batch_gd = True
                    break
                else:
                    theta = potential_theta
                    diff = current_value - potential_value
                    current_value = potential_value
                    print "diff", diff
                    if diff < diff_thres:
                        # Stop if difference in objective function is too small
                        if do_batch_gd:
                            finish_gd = True
                        do_batch_gd = True
                        break
            print "step size", step_size
            if do_batch_gd:
                print "do batch gd"
                step_size = init_step_size
                prox_grad_batch_size = self.num_samples
            if finish_gd:
                break
        log.info("final GD iter %d, val %f, time %d" % (i, current_value, time.time() - st))
        # prox: final GD iter 102, val 6.092593, time 28
        # old:
        1/0
        return theta, -current_value

    def _get_gradient_log_lik(self, theta, feat_vec_sample_pair_batch):
        """
        @param theta: where to take the gradient of the total log likelihood

        @return the gradient of the total log likelihood wrt theta
        """
        grads = [self._get_gradient_log_lik_per_sample(theta, sample, feature_vecs) for sample, feature_vecs in feat_vec_sample_pair_batch]
        grad_ll_dtheta = np.sum(grads, axis=0)
        return -1.0/self.num_samples * grad_ll_dtheta

    def _get_gradient_log_lik_per_sample(self, theta, sample, feature_vecs):
        """
        @param theta: where to take the gradient of the total log likelihood
        @param sample: class ImputedSequenceMutations
        @param feature_vecs: list of sparse feature vectors for all at-risk positions at every mutation step

        @return the gradient of the log likelihood for this sample
        """
        grad = np.zeros(theta.size)
        for mutating_pos, vecs_at_mutation_step in zip(sample.mutation_order, feature_vecs):
            grad[vecs_at_mutation_step[mutating_pos]] += 1
            denom = 0
            grad_log_sum_exp = np.zeros(theta.size)
            for one_feats in vecs_at_mutation_step.values():
                val = np.exp(np.sum(theta[one_feats]))
                for f in one_feats:
                    grad_log_sum_exp[f] += val
                    denom += val
            grad -= grad_log_sum_exp/denom
        return grad
