import numpy as np

class MethodResults:
    def __init__(self, penalty_params, theta, fitted_prob_vector):
        self.penalty_params = penalty_params
        self.theta = theta
        self.fitted_prob_vector = fitted_prob_vector

    def __str__(self):
        pen_param_str = ",".join(map(str, self.penalty_params))
        num_nonzero = np.sum(self.theta > 1e-6)
        return "Pen params %s, nonzero theta %d" % (pen_param_str, num_nonzero)
