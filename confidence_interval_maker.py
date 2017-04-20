import numpy as np
import scipy as sp

class ConfidenceIntervalMaker:
    def run(self, theta, e_step_samples, problem, num_e_sample_per_obs):
        sample_obs_information, sample_hessian, _ = problem.get_hessian(theta)

        sample_hessian = 1.0/num_e_sample_per_obs * sample_hessian
        variance_est = np.linalg.pinv(sample_hessian)
        print "sample_hessian np.diag(variance_est)", np.diag(variance_est)
        # std_error = np.sqrt(np.diag(variance_est))
        # print "std_error", std_error

        sample_obs_information = 1.0/num_e_sample_per_obs * sample_obs_information
        variance_est = np.linalg.pinv(sample_obs_information)
        print "sample_obs_information variance_est", np.diag(variance_est)
        # std_error = np.sqrt(np.diag(variance_est))
        # print "std_error", std_error
