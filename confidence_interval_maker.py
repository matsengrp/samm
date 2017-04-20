import numpy as np
import scipy as sp

class ConfidenceIntervalMaker:
    def run(self, theta, e_step_samples, problem, num_e_sample_per_obs):
        sample_obs_information, _ = problem.get_hessian(theta)

        print "sample_obs_information eigvals", np.linalg.eigvals(sample_obs_information)
        variance_est = np.linalg.inv(sample_obs_information)
