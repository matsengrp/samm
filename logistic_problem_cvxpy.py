import cvxpy as cp
import numpy as np

from common import NUM_NUCLEOTIDES

def _logistic(z):
    return np.log(1 + np.exp(z))

class LogisticRegressionMotif:
    def __init__(self, theta_shape, X, y, init_lam=1, per_target_model=False):
        self.per_target_model = per_target_model
        self.theta_intercept = cp.Variable()
        self.theta = cp.Variable(theta_shape)
        theta_norm = cp.norm(self.theta, 1)
        self.lam = cp.Parameter(nonneg=True, value=init_lam)

        log_ll = -cp.sum(cp.logistic(-(X * (self.theta[:,0:1] + self.theta_intercept))))

        no_mutate_X = X[y == 0, :]
        no_mutate_numerator = - (no_mutate_X * (self.theta[:,0:1] + self.theta_intercept))

        log_ll = log_ll + cp.sum(no_mutate_numerator)
        if per_target_model:
            for i in range(NUM_NUCLEOTIDES):
                mutate_X_targ = X[y == (i + 1), :]
                target_ll = -cp.logistic(- (mutate_X_targ * (self.theta[:,i + 1] + self.theta_intercept)))
                log_ll += cp.sum(target_ll)

        self.problem = cp.Problem(cp.Maximize(log_ll - self.lam * theta_norm))

    def solve(self, lam_val, max_iters=2000, verbose=False):
        self.lam.value = lam_val
        self.problem.solve(max_iters=max_iters, verbose=verbose)
        assert(self.problem.status == cp.OPTIMAL)
        return self.theta.value, self.theta.value + self.theta_intercept.value, self.problem.value

    def score(self, new_X, new_y):
        log_ll = - np.sum(_logistic(-new_X.dot(self.theta.value[:,0:1] + self.theta_intercept.value)))

        no_mutate_X = new_X[new_y == 0, :]
        no_mutate_numerator = - (no_mutate_X.dot(self.theta.value[:,0:1] + self.theta_intercept.value))

        log_ll = log_ll + np.sum(no_mutate_numerator)
        if self.per_target_model:
            for i in range(NUM_NUCLEOTIDES):
                mutate_X_targ = new_X[new_y == (i + 1), :]
                target_ll = - _logistic(- (mutate_X_targ.dot(self.theta.value[:,i+1:i+2] + self.theta_intercept.value)))
                log_ll += np.sum(target_ll)
        return log_ll
