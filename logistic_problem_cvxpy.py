import cvxpy as cp
import numpy as np

from common import NUM_NUCLEOTIDES

def _logistic(z):
    return np.log(1 + np.exp(z))

class LogisticRegressionMotif:
    """
    Fit logistic regression with L1 penalty on the theta matrix via cvxpy
    (We use cvxpy because the sparsity pattern is very specific for per-target and hierarchical models)

    The probability of mutating follows usual logistic regression parameterization
    The probability of substituting particular nucleotides follows a softmax parameterization
    """
    def __init__(self, theta_shape, X, y, y_orig, init_lam=1, per_target_model=False):
        self.X = X
        self.y = y
        self.y_orig = y_orig

        self.per_target_model = per_target_model
        self.theta_intercept = cp.Variable()
        self.theta = cp.Variable(theta_shape[0], theta_shape[1])
        theta_norm = cp.norm(self.theta, 1)
        self.lam = cp.Parameter(sign="positive", value=init_lam)

        # This is the log denominator of the probability of mutating (-log(1 + exp(-theta)))
        log_ll = -cp.sum_entries(cp.logistic(-(X * (self.theta[:,0:1] + self.theta_intercept))))

        # If no mutation happened, then we also need the log numerator of probability of not mutating
        # since exp(-theta)/(1 + exp(-theta)) is prob not mutate
        no_mutate_X = X[y == 0, :]
        no_mutate_numerator = - (no_mutate_X * (self.theta[:,0:1] + self.theta_intercept))

        log_ll = log_ll + cp.sum_entries(no_mutate_numerator)
        if per_target_model:
            # If per target, need the substitution probabilities too
            for orig_i in range(NUM_NUCLEOTIDES):
                for i in range(NUM_NUCLEOTIDES):
                    if orig_i == i:
                        continue

                    # Find the elements that mutated to y and mutated from y_orig
                    mutate_X_targ = X[(y == (i + 1)) & (y_orig == (orig_i + 1)), :]
                    # Create the 3 column theta excluding the column corresponding to y_orig
                    theta_3col = []
                    for j in range(NUM_NUCLEOTIDES):
                        if j != orig_i:
                            theta_3col.append(self.theta[:,j + 1] + self.theta_intercept)

                    theta_3col = cp.hstack(theta_3col)
                    target_ll = (
                            # log of numerator in softmax
                            - (mutate_X_targ * (self.theta[:,i + 1] + self.theta_intercept))
                            # log of denominator in softmax
                            - cp.log_sum_exp(-(mutate_X_targ * theta_3col), axis=1))
                    log_ll += cp.sum_entries(target_ll)

        self.problem = cp.Problem(cp.Maximize(log_ll - self.lam * theta_norm))

    def solve(self, lam_val, max_iters=2000, verbose=False):
        self.lam.value = lam_val
        self.problem.solve(max_iters=max_iters, verbose=verbose)
        assert(self.problem.status == cp.OPTIMAL)
        return np.array(self.theta.value), np.array(self.theta.value + self.theta_intercept.value), self.problem.value

    def score(self, new_X, new_y, new_y_orig):
        """
        @return log likelihood on this new dataset
        """
        # This code is the numpy version of the stuff in __init__
        log_ll = - np.sum(_logistic(-new_X.dot(self.theta.value[:,0:1] + self.theta_intercept.value)))

        no_mutate_X = new_X[new_y == 0, :]
        no_mutate_numerator = - (no_mutate_X.dot(self.theta.value[:,0:1] + self.theta_intercept.value))

        log_ll = log_ll + np.sum(no_mutate_numerator)
        if self.per_target_model:
            for orig_i in range(NUM_NUCLEOTIDES):
                for i in range(NUM_NUCLEOTIDES):
                    if orig_i == i:
                        continue

                    # Find the elements that mutated to y and mutated from y_orig
                    mutate_X_targ = new_X[(new_y == (i + 1)) & (new_y_orig == (orig_i + 1)), :]
                    # Create the 3 column theta excluding the column corresponding to y_orig
                    theta_3col = []
                    for j in range(NUM_NUCLEOTIDES):
                        if j != orig_i:
                            theta_3col.append(self.theta.value[:,j + 1] + self.theta_intercept.value)

                    theta_3col = np.hstack(theta_3col)
                    target_ll = (
                            # log of numerator in softmax
                            - mutate_X_targ.dot(self.theta.value[:,i + 1:i + 2] + self.theta_intercept.value)
                            # log of denominator in softmax
                            - np.log(np.sum(np.exp(-(mutate_X_targ.dot(theta_3col))), axis=1)))
                    log_ll += np.sum(target_ll)
        return log_ll
