import time
from numpy import zeros
from cvxpy import *
from feature_generator import FeatureGenerator

class SurvivalProblem:
    def __init__(self, samples, feature_generator):
        self.samples = samples
        self.feature_generator = feature_generator

    def solve(self, lasso_param, verbose=False):
        # TODO: Add theta for different mutation types
        theta = Variable(self.feature_generator.feature_vec_len)
        obj = 0
        for sample in self.samples:
            obj += self.calculate_per_sample_log_lik(theta, sample)
        # maximize the average log likelihood (normalization makes it easier to track EM
        # since the number of E-step samples grows)
        problem = Problem(Maximize(1.0/len(self.samples) * obj - lasso_param * norm(theta, 1)))
        problem.solve(verbose=verbose)
        assert(problem.status == OPTIMAL)
        return theta.value, problem.value

    def calculate_log_lik_vec(self, theta, prev_theta):
        log_lik_vec = zeros(len(self.samples))
        for sample_id, sample in enumerate(self.samples):
            log_lik_vec[sample_id] = self.calculate_per_sample_log_lik(theta, sample).value - \
                    self.calculate_per_sample_log_lik(prev_theta, sample).value
        return log_lik_vec

    def calculate_per_sample_log_lik(self, theta, sample):
        all_feature_vecs = self.feature_generator.create_for_mutation_steps(sample)
        obj = 0
        for mutating_pos, vecs_at_mutation_step in zip(sample.mutation_order, all_feature_vecs):
            # vec_mutation_step are the feature vectors of the at-risk group after mutation i
            feature_vec_mutated = vecs_at_mutation_step[mutating_pos]
            obj += sum_entries(theta[feature_vec_mutated]) - log_sum_exp(vstack(*[
                sum_entries(theta[f]) for f in vecs_at_mutation_step.values()
            ]))
        return obj

