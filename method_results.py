class MethodResults:
    def __init__(self, penalty_params, theta, fitted_prob_vector, val_log_lik):
        self.penalty_params = penalty_params
        self.theta = theta
        self.fitted_prob_vector = fitted_prob_vector
        self.val_log_lik = val_log_lik
