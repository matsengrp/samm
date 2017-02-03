import time
from numpy import zeros
from cvxpy import *
from common import *
from survival_problem import SurvivalProblem

class SurvivalProblemMultiCVXPY(SurvivalProblem):
    """
    Use CVXPY to solve the (penalized) survival problem
    multiple thetas

    Warning: will be very slow for large datasets
    """
    def __init__(self, feature_generator, samples, penalty_param, theta_mask):
        """
        @param samples: the observations for this problem, list of ImputedSequenceMutations
        @param feature_generator: FeatureGenerator
        @param penalty_param: the coefficient on the penalty function(s). This assumes a single
                                shared penalty parameter across all penalties for now.
        """
        assert(penalty_param >= 0)

        self.samples = samples
        self.feature_generator = feature_generator
        self.penalty_param = penalty_param
        motif_list = feature_generator.get_motif_list()
        self.theta_mask = theta_mask

    def calculate_log_lik_ratio_vec(self, theta, prev_theta):
        log_lik_vec = zeros(len(self.samples))
        for sample_id, sample in enumerate(self.samples):
            log_lik_vec[sample_id] = self.calculate_per_sample_log_lik(theta, sample).value - \
                    self.calculate_per_sample_log_lik(prev_theta, sample).value
        return log_lik_vec

    def calculate_per_sample_log_lik(self, theta, sample):
        all_feature_vecs, _ = self.feature_generator.create_for_mutation_steps(sample)
        obj = Parameter(1, value=0)
        for mutating_pos, vecs_at_mutation_step in zip(sample.mutation_order, all_feature_vecs):
            # vec_mutation_step are the feature vectors of the at-risk group after mutation i
            feature_vec_mutated = vecs_at_mutation_step[mutating_pos]
            target_nucleotide = NUCLEOTIDE_DICT[sample.obs_seq_mutation.end_seq[mutating_pos]]

            obj += sum_entries(theta[feature_vec_mutated, target_nucleotide]) - log_sum_exp(vstack(*[
                sum_entries(theta[f, i])
                for f in vecs_at_mutation_step.values()
                for i in range(NUM_NUCLEOTIDES)
                if self.theta_mask[f,i]
            ]))
        return obj

class SurvivalProblemLassoMultiCVXPY(SurvivalProblemMultiCVXPY):
    """
    Objective function: log likelihood of theta - lasso penalty on theta
    * Lasso penalty over all of theta

    Note: motifs that mutate to different target nucleotides share the same theta value
    """
    def get_value(self, theta):
        obj = 0
        for sample in self.samples:
            obj += self.calculate_per_sample_log_lik(theta, sample)
        # maximize the average log likelihood (normalization makes it easier to track EM
        # since the number of E-step samples grows)
        penalized_obj = 1.0/len(self.samples) * obj - self.penalty_param * norm(theta[self.theta_mask], 1)
        return penalized_obj

    def solve(self, init_theta=None, max_iters=None, num_threads=1):
        """
        @param init_theta: use this to figure out which thetas are not supposed to be estimated (since they are -inf)
        @param max_iters: ignored
        @param num_threads: ignored
        """
        theta = Variable(self.feature_generator.feature_vec_len, NUM_NUCLEOTIDES)
        # maximize the average log likelihood
        problem = Problem(
            Maximize(self.get_value(theta))
        )
        problem.solve(verbose=True, solver=SCS)
        assert(problem.status == OPTIMAL)
        return theta.value, problem.value
