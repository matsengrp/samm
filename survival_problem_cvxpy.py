import time
from numpy import zeros
from cvxpy import *
from common import *
from survival_problem import SurvivalProblem

class SurvivalProblemCVXPY(SurvivalProblem):
    """
    Use CVXPY to solve the (penalized) survival problem

    Warning: will be very slow for large datasets
    """
    def calculate_log_lik_ratio_vec(self, theta, prev_theta, group_by_sample=False):
        assert(not group_by_sample)
        log_lik_vec = zeros(len(self.samples))
        for sample_id, sample in enumerate(self.samples):
            log_lik_vec[sample_id] = self.calculate_per_sample_log_lik(theta, sample).value - \
                    self.calculate_per_sample_log_lik(prev_theta, sample).value
        return log_lik_vec

    def calculate_per_sample_log_lik(self, theta, sample):
        obj = Parameter(1, value=0)

        seq_str = sample.obs_seq_mutation.start_seq
        for i in range(len(sample.mutation_order)):
            mutating_pos = sample.mutation_order[i]

            feature_dict = self.feature_generator.create_for_sequence(
                seq_str,
                sample.obs_seq_mutation.left_flank,
                sample.obs_seq_mutation.right_flank,
                set(range(sample.obs_seq_mutation.seq_len)) - set(sample.mutation_order[:i])
            )
            # vecs_at_mutation_step[i] are the feature vectors of the at-risk group after mutation i
            feature_idx_mutated = feature_dict[mutating_pos]
            col_idx = 0
            if self.per_target_model:
                col_idx = NUCLEOTIDE_DICT[sample.obs_seq_mutation.end_seq[mutating_pos]]

            obj += sum_entries(theta[feature_idx_mutated, col_idx]) - log_sum_exp(vstack(*[
                sum_entries(theta[feat_idx, i])
                for feat_idx in feature_dict.values()
                for i in range(self.theta_num_col)
                if self.theta_mask[feat_idx,i]
            ]))

            seq_str = mutate_string(
                seq_str,
                mutating_pos,
                sample.obs_seq_mutation.end_seq[mutating_pos]
            )
        return obj

    def get_log_likelihood(self, theta):
        obj = 0
        for sample in self.samples:
            obj += self.calculate_per_sample_log_lik(theta, sample)
        # maximize the average log likelihood (normalization makes it easier to track EM
        # since the number of E-step samples grows)
        return 1.0/len(self.samples) * obj

class SurvivalProblemLassoCVXPY(SurvivalProblemCVXPY):
    """
    Objective function: log likelihood of theta - lasso penalty on theta
    * Lasso penalty over all of theta

    Note: motifs that mutate to different target nucleotides share the same theta value
    """
    def get_value(self, theta):
        ll = self.get_log_likelihood(theta)
        return ll - self.penalty_param * norm(theta[self.theta_mask], 1)

    def solve(self, init_theta=None, max_iters=None, num_threads=1):
        """
        @param init_theta: only used to determine shape of theta
        @param max_iters: ignored
        @param num_threads: ignored
        """
        theta = Variable(init_theta.shape[0], init_theta.shape[1])

        # maximize the average log likelihood (normalization makes it easier to track EM
        # since the number of E-step samples grows)
        problem = Problem(Maximize(self.get_value(theta)))
        problem.solve(solver=SCS, verbose=True)
        assert(problem.status == OPTIMAL)
        return theta.value, problem.value

class SurvivalProblemLassoCVXPY_ADMM(SurvivalProblemCVXPY):
    """
    Objective function: log likelihood of theta - lasso penalty on theta - augmented penalty on theta from ADMM
    * Lasso penalty over all of theta
    * || difference in thetas - residuals ||^2 from ADMM

    Internal solver for ADMM for comparison to our grad descent implementation

    Note: motifs that mutate to different target nucleotides share the same theta value
    """
    def solve(self, beta,u, D, init_theta=None, max_iters=None, num_threads=1):
        """
        @param init_theta: ignored
        @param max_iters: ignored
        @param num_threads: ignored
        """
        theta = Variable(self.feature_generator.feature_vec_len, 1)
        obj = 0
        for sample in self.samples:
            obj += self.calculate_per_sample_log_lik(theta, sample)
        # maximize the average log likelihood (normalization makes it easier to track EM
        # since the number of E-step samples grows)
        problem = Problem(Minimize(
            - 1.0/len(self.samples) * obj
            + self.penalty_param * norm(theta, 1)
            + 0.5 * pow(norm(beta - D * theta + u, 2), 2)
        ))
        problem.solve(verbose=True)
        assert(problem.status == OPTIMAL)
        return theta.value, problem.value

class SurvivalProblemFusedLassoCVXPY(SurvivalProblemCVXPY):
    """
    Objective function: log likelihood of theta - penalty_param * (fused lasso penalty + lasso penalty)
    * Creates fused lasso penalties between motifs that differ by only a single nucleotide
    * Lasso penalty over all of theta

    Note: motifs that mutate to different target nucleotides share the same theta value
    """
    def solve(self, init_theta=None, max_iters=None, num_threads=1):
        """
        @param init_theta: ignored
        @param max_iters: ignored
        @param num_threads: ignored
        """
        motif_list = self.feature_generator.motif_list

        theta = Variable(self.feature_generator.feature_vec_len)
        obj = 0
        for sample in self.samples:
            obj += self.calculate_per_sample_log_lik(theta, sample)

        fused_lasso_pen = 0
        for i1, m1 in enumerate(motif_list):
            for i2, m2 in enumerate(motif_list):
                if i1 == i2:
                    continue
                idx_differ = get_idx_differ_by_one_character(m1, m2)
                if idx_differ is None:
                    continue
                else:
                    fused_lasso_pen += abs(theta[i1] - theta[i2])

        problem = Problem(Maximize(1.0/len(self.samples) * obj - self.penalty_param * fused_lasso_pen - self.penalty_param * norm(theta,1)))
        problem.solve(solver=SCS, verbose=True)
        assert(problem.status == OPTIMAL)
        return theta.value, problem.value
