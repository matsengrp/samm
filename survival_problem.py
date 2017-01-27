class SurvivalProblem:
    def __init__(self, feature_generator, samples, penalty_param):
        """
        @param samples: the observations for this problem
        @param feature_generator: FeatureGenerator
        @param penalty_param: the coefficient on the penalty function(s). This assumes a single
                                shared penalty parameter across all penalties for now.
        """
        assert(penalty_param >= 0)

        self.samples = samples
        self.feature_generator = feature_generator
        self.penalty_param = penalty_param

    def solve(self, init_theta=None, max_iters=None, num_threads=1):
        """
        Solve the problem
        @param init_theta: if this param is used, then the problem solver will initialize the theta
                            at this value
        @param max_iters: if this param is used, then this is the maximum number of iterations for
                            the problem solver
        @param num_threads: if this param is used, number of threads to use to calculate the gradient

        @return final fitted value of theta and objective function value
        """
        raise NotImplementedError()

    def calculate_log_lik_ratio_vec(self, theta, prev_theta):
        """
        @param theta: the theta in the numerator
        @param prev_theta: the theta in the denominator
        @return the log likelihood ratios between theta and prev_theta for each e-step sample
        """
        raise NotImplementedError()
