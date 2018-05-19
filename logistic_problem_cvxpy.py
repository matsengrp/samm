import cvxpy as cp
import numpy as np

from common import NUM_NUCLEOTIDES

class LogisticRegressionMotif:
    def __init__(self, theta_shape, X, y, init_lam=1, per_target_model=False):
        self.theta = cp.Variable(theta_shape)
        theta_norm = cp.norm(self.theta, 1)
        self.lam = cp.Parameter(nonneg=True, value=init_lam)

        num_no_mutate = np.sum(y == 0)
        mutate_ll = -cp.logistic(-(X * self.theta[:,0]))

        no_mutate_X = X[y == 0, :]
        no_mutate_numerator = - (no_mutate_X * self.theta[:,0])

        log_ll = cp.sum(mutate_ll) + cp.sum(no_mutate_numerator)
        if per_target_model:
            for i in range(NUM_NUCLEOTIDES):
                mutate_X_targ = X[y == (i + 1), :]
                target_ll = -cp.logistic(- (mutate_X_targ * self.theta[:,i + 1]))
                log_ll += cp.sum(target_ll)

        self.problem = cp.Problem(cp.Maximize(log_ll - self.lam * theta_norm))

    def solve(self, max_iters=2000):
        self.problem.solve(max_iters=max_iters, verbose=True)
        assert(self.problem.status == cp.OPTIMAL)
        return self.theta.value, self.problem.value
