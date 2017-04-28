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

    def _get_standard_error(self, information_matrix):
        eigenvals = np.linalg.eigvals(information_matrix)
        if np.all(np.abs(eigenvals) > 0):
            variance_est = np.linalg.inv(information_matrix)
            variance_est = np.diag(variance_est)
            # Make sure that the variance estimate makes sense -- should be positive values only
            if np.all(variance_est > 0):
                standard_errors = np.sqrt(variance_est)
                return eigenvals, variance_est, standard_errors
            else:
                return eigenvals, variance_est, None
        else:
            return eigenvals, None, None

    def _print_standard_error_est(self, eigenvals, variance_est, se_est, print_prefix=""):
        log.info("%s eigvals %f" % (print_prefix, np.min(np.abs(eigenvals))))
        if variance_est is not None:
            log.info("%s variance est %f, %f, %f" % (print_prefix, np.median(variance_est), np.max(variance_est), np.min(variance_est)))
        if se_est is not None:
            log.info("%s std error est %f, %f, %f" % (print_prefix, np.median(se_est), np.max(se_est), np.min(se_est)))

    def run(self, theta, e_step_samples, problem, z=1.96):
        """
        The asymptotic covariance matrix of theta is the inverse of the fisher's information matrix
        since theta is an MLE.

        @param z: the z-statistic that controls the width of the confidence intervals

        @return standard error estimates for the theta parameters
        """
        log.info("Obtaining Confidence Interval Estimates...")
        obs_fisher_info, complete_fisher_info, _ = problem.get_hessian(theta)
        complete_fisher_info = (complete_fisher_info[self.theta_mask_flat,:])[:,self.theta_mask_flat]
        complete_eigenvals, complete_variance_est, complete_standard_errors = self._get_standard_error(complete_fisher_info)
        self._print_standard_error_est(complete_eigenvals, complete_variance_est, complete_standard_errors, print_prefix="Complete")
        is_complete_fisher_ok = complete_standard_errors is not None

        obs_fisher_info = (obs_fisher_info[self.theta_mask_flat,:])[:,self.theta_mask_flat]
        obs_eigenvals, obs_variance_est, obs_standard_errors = self._get_standard_error(obs_fisher_info)
        self._print_standard_error_est(obs_eigenvals, obs_variance_est, obs_standard_errors, print_prefix="Observed")

        if obs_standard_errors is not None:
            conf_ints = self._create_confidence_intervals(obs_standard_errors, theta, z)
            log.info(self._get_confidence_interval_print_lines(conf_ints))
            return obs_standard_errors, is_complete_fisher_ok

        return None, is_complete_fisher_ok

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
