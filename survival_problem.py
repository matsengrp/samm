from cvxpy import *
from feature_generator import FeatureGenerator

class SurvivalProblem:
    def __init__(self, samples):
        self.samples = samples

    def solve(self, feature_generator):
        # TODO: Add theta for different mutation types
        theta = Variable(feature_generator.feature_vec_len)
        obj = 0
        for sample in self.samples:
            obj += self.calculate_per_sample_lik(feature_generator, theta, sample)
        # maximize the average log likelihood (normalization makes it easier to track EM
        # since the number of E-step samples grows)
        problem = Problem(Maximize(1.0/len(self.samples) * obj))
        problem.solve()
        assert(problem.status == OPTIMAL)
        return theta.value, problem.value

    def calculate_lik_stats(self, feature_generator, theta, prev_theta, sample):
        lik_mean = 0.
        lik_var = 0.
        n = len(self.samples)
        for sample_id, sample in enumerate(self.samples):
            lik_value = self.calculate_per_sample_lik(feature_generator, theta, sample) - \
                    self.calculate_per_sample_lik(feature_generator, prev_theta, sample)
            lik_mean += lik_value / n
            lik_var += lik_value*lik_value / n
        lik_var -= lik_mean*lik_mean
        return lik_mean.value, lik_var.value

    def calculate_per_sample_lik(self, feature_generator, theta, sample):
        all_feature_vecs = feature_generator.create_for_mutation_steps(sample)
        obj = 0
        for mutating_pos, vecs_at_mutation_step in zip(sample.mutation_order, all_feature_vecs):
            # vec_mutation_step are the feature vectors of the at-risk group after mutation i
            feature_vec_mutated = vecs_at_mutation_step[mutating_pos]
            obj += sum_entries(theta[feature_vec_mutated]) - log_sum_exp(vstack(*[
                sum_entries(theta[f]) for f in vecs_at_mutation_step.values()
            ]))
        return obj

