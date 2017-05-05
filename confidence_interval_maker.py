import numpy as np
import scipy as sp
import logging as log

from common import NUCLEOTIDES

class ConfidenceIntervalMaker:
    def __init__(self, motif_list, per_target_model, possible_theta_mask, zero_theta_mask):
        self.motif_list = motif_list
        self.per_target_model = per_target_model
        self.possible_theta_mask = possible_theta_mask
        self.zero_theta_mask = zero_theta_mask
        self.theta_mask = possible_theta_mask & ~zero_theta_mask
        self.theta_mask_flat = self.theta_mask.reshape((self.theta_mask.size,), order="F")

    def run(self, theta, e_step_samples, problem, z=1.96):
        """
        The asymptotic covariance matrix of theta is the inverse of the fisher's information matrix
        since theta is an MLE.

        @param z: the z-statistic that controls the width of the confidence intervals

        @return standard error estimates for the theta parameters
        """
        log.info("Obtaining Confidence Interval Estimates...")
        sample_obs_information, _ = problem.get_hessian(theta)
        # Need to filter out all the theta values that are constant (negative infinity or zero constants)
        sample_obs_information = (sample_obs_information[self.theta_mask_flat,:])[:,self.theta_mask_flat]

        # Make sure that we can take an inverse -- need eigenvalues to be nonzero
        eigenvals = np.linalg.eigvals(sample_obs_information)
        log.info("np.linalg.eigvals %s" % eigenvals)
        if np.all(np.abs(np.linalg.eigvals(sample_obs_information)) > 0):
            variance_est = np.linalg.inv(sample_obs_information)

            # Make sure that the variance estimate makes sense -- should be positive values only
            if np.all(np.diag(variance_est) > 0):
                standard_errors = np.sqrt(np.diag(variance_est))
                conf_ints = self._create_confidence_intervals(standard_errors, theta, z)
                log.info(self._get_confidence_interval_print_lines(conf_ints))
                return variance_est
            else:
                log.info("Variance estimates are negative %s" % np.diag(variance_est))
        else:
            log.info("Confidence interval: observation matrix is singular")
        return None

    def _create_confidence_intervals(self, standard_errors, theta, z=1.96):
        """
        Creates the confidence intervals using the normal approximation
        """
        theta_flat = theta.reshape((theta.size,), order="F")
        theta_flat = theta_flat[self.theta_mask_flat]
        conf_int_low = theta_flat - z * standard_errors
        conf_int_upper = theta_flat + z * standard_errors
        conf_ints = np.hstack((
            conf_int_low.reshape((conf_int_low.size, 1)),
            theta_flat.reshape((theta_flat.size, 1)),
            conf_int_upper.reshape((conf_int_upper.size, 1)),
        ))
        return conf_ints

    def _get_confidence_interval_print_lines(self, conf_ints):
        """
        Get confidence intervals print lines (shows motif and target nucleotides)
        Sorted by theta values
        """
        print_line_list = []
        idx = 0
        for i in range(self.zero_theta_mask.shape[0]):
            motif = self.motif_list[i]
            for j in range(self.zero_theta_mask.shape[1]):
                target_nucleotide = "n" if j == 0 else NUCLEOTIDES[j - 1]
                if not self.theta_mask[i, j]:
                    continue
                print_str = "%s (%s->%s)" % (conf_ints[idx,:], motif, target_nucleotide)
                print_line_list.append((conf_ints[idx,1], print_str))
                idx += 1
        sorted_lines = sorted(print_line_list, key=lambda s: s[0])
        return "\n".join([l[1] for l in sorted_lines])
