import numpy as np
import scipy as sp

class ConfidenceIntervalMaker:
    def run(self, theta, e_step_samples, problem, num_e_sample_per_obs):
        sample_obs_information, _ = problem.get_hessian(theta)

        print "sample_obs_information eigvals", np.linalg.eigvals(sample_obs_information)
        if np.all(np.abs(np.linalg.eigvals(sample_obs_information))) > 0:
            variance_est = np.linalg.inv(sample_obs_information)
            print "np.diag(variance_est)", np.diag(variance_est)
            if np.all(np.diag(variance_est)) > 0:
                print "standard errors?", np.sqrt(np.diag(variance_est))
