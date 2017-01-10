from cvxpy import *
from feature_generator import FeatureGenerator

class SurvivalProblem:
    def __init__(self, samples):
        self.samples = samples

    def solve(self, feature_generator):
        theta = Variable(feature_generator.feature_vec_len)
        obj = 0
        for sample in self.samples:
            all_feature_vecs = feature_generator.create_for_mutation_steps(sample)
            for mutating_pos, vecs_at_mutation_step in zip(sample.mutation_order, all_feature_vecs):
                # vec_mutation_step are the feature vectors of the at-risk group after mutation i
                feature_vec_mutated = vecs_at_mutation_step[mutating_pos]
                obj += sum_entries(theta[feature_vec_mutated]) - log_sum_exp(vstack(*[
                    sum_entries(theta[f]) for f in vecs_at_mutation_step.values()
                ]))
        problem = Problem(Maximize(obj))
        problem.solve()
        assert(problem.status == OPTIMAL)
        return theta.value
