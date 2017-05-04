import numpy as np
from common import *

class MethodResults:
    def __init__(self, penalty_params, theta, fitted_prob_vector, variance_est=None):
        self.penalty_params = penalty_params
        self.theta = theta
        self.fitted_prob_vector = fitted_prob_vector
        self.variance_est = variance_est
        self.num_nonzero = get_num_nonzero(self.theta)
        self.num_unique = get_num_unique_theta(self.theta)

    def __str__(self):
        pen_param_str = ",".join(map(str, self.penalty_params))
        return "Pen params %s, nonzero %d, unique %d" % (pen_param_str, self.num_nonzero, self.num_unique)
