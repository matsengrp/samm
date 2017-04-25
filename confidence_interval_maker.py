import numpy as np
import scipy as sp

class ConfidenceIntervalMaker:
    def run(self, theta, e_step_samples, problem):
        sample_obs_information, _ = problem.get_hessian(theta)

        print "sample_obs_information eigvals", np.linalg.eigvals(sample_obs_information)
        if np.all(np.abs(np.linalg.eigvals(sample_obs_information)) > 0):
            variance_est = np.linalg.inv(sample_obs_information)
            print "np.diag(variance_est)", np.diag(variance_est)
            if np.all(np.diag(variance_est) > 0):
                print "standard errors?", np.sqrt(np.diag(variance_est))
                standard_errors = np.sqrt(np.diag(variance_est))
                theta_flat = theta.reshape((theta.size,), order="F")
                print "standard_errors", standard_errors.shape
                conf_int_low = theta_flat - 1.96 * standard_errors
                print "conf_int_low", conf_int_low.shape
                conf_int_upper = theta_flat + 1.96 * standard_errors
                print "conf_ints", np.hstack((
                    conf_int_low.reshape((conf_int_low.size, 1)),
                    theta_flat.reshape((theta_flat, 1)),
                    conf_int_upper.reshape((conf_int_upper.size, 1)),
                ))
                return standard_errors
        else:
            return None
