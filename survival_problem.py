from common import NUM_NUCLEOTIDES

class SurvivalProblem:
    def solve(self, init_theta=None, max_iters=None):
        """
        Solve the problem
        @param init_theta: if this param is used, then the problem solver will initialize the theta
                            at this value
        @param max_iters: if this param is used, then this is the maximum number of iterations for
                            the problem solver

        @return final fitted value of theta and objective function value
        """
        raise NotImplementedError()

    def calculate_log_lik_ratio_vec(self, theta, prev_theta, group_by_sample=False):
        """
        @param theta: the theta in the numerator
        @param prev_theta: the theta in the denominator
        @param group_by_sample: whether to group the log lik ratio values by sample
                                (so if we get 4 e-samples, this returns 4 values)
        @return the log likelihood ratios between theta and prev_theta for each e-step sample
        """
        raise NotImplementedError()
