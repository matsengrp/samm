from common import NUM_NUCLEOTIDES

class SurvivalProblem:
    def __init__(self, feature_generator, samples, penalty_param, theta_mask, num_threads=1):
        """
        @param samples: the observations for this problem, list of ImputedSequenceMutations
        @param feature_generator: FeatureGenerator
        @param penalty_param: the coefficient on the penalty function(s). This assumes a single
                                shared penalty parameter across all penalties for now.
        @param theta_mask: a mask indicating which theta values to estimate
        """
        assert(penalty_param >= 0)

        self.samples = samples
        self.feature_generator = feature_generator
        self.penalty_param = penalty_param
        self.theta_mask = theta_mask
        self.theta_num_col = self.theta_mask.shape[1]
        self.per_target_model = self.theta_num_col == NUM_NUCLEOTIDES
        self.num_threads = num_threads

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

    def calculate_log_lik_ratio_vec(self, theta, prev_theta):
        """
        @param theta: the theta in the numerator
        @param prev_theta: the theta in the denominator
        @return the log likelihood ratios between theta and prev_theta for each e-step sample
        """
        raise NotImplementedError()
