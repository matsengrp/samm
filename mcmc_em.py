import time
import numpy as np
import logging as log

from models import *
from common import *
from sampler_collection import SamplerCollection

class MCMC_EM:
    def __init__(self, observed_data, feat_generator, sampler_cls, problem_solver_cls, base_num_e_samples=10, burn_in=10, max_m_iters=500, num_threads=1):
        """
        @param observed_data: list of observed data (start and end sequences)
        @param feat_generator: an instance of a FeatureGenerator
        @param sampler_cls: a Sampler class
        @param problem_solver_cls: SurvivalProblem class
        @param base_num_e_samples: number of E-step samples to draw initially
        @param burn_in: number of gibbs sweeps to do for burn in
        @param max_m_iters: maximum number of iterations for the M-step
        @param num_threads: number of threads to use
        """
        self.observed_data = observed_data
        self.feat_generator = feat_generator
        self.base_num_e_samples = base_num_e_samples
        self.max_m_iters = max_m_iters
        self.burn_in = burn_in
        self.sampler_cls = sampler_cls
        self.problem_solver_cls = problem_solver_cls
        self.num_threads = num_threads

    def run(self, theta=None, penalty_param=1, max_em_iters=10, diff_thres=1e-6, max_e_samples=1000):
        """
        @param theta: initial value for theta in MCMC-EM
        @param penalty_param: the coefficient for the penalty function
        @param max_em_iters: the maximum number of iterations of MCMC-EM
        @param diff_thres: if the change in the objective function changes no more than `diff_thres`, stop MCMC-EM
        """
        st = time.time()
        # initialize theta vector
        if theta is None:
            theta = np.random.randn(self.feat_generator.feature_vec_len)
        # stores the initialization for the gibbs samplers for the next iteration's e-step
        init_orders = [obs_seq.mutation_pos_dict.keys() for obs_seq in self.observed_data]
        prev_pen_exp_log_lik = None
        for run in range(max_em_iters):
            lower_bound_is_negative = True
            prev_theta = theta
            num_e_samples = self.base_num_e_samples
            burn_in = self.burn_in

            sampler_collection = SamplerCollection(
                self.observed_data,
                prev_theta,
                self.sampler_cls,
                self.feat_generator,
                self.num_threads,
            )

            e_step_samples = []
            while lower_bound_is_negative and len(e_step_samples) < max_e_samples:
                ## Keep grabbing samples until it is highly likely we have increased the penalized log likelihood

                # do E-step
                log.info("E STEP, iter %d, num samples %d, time %f" % (run, len(e_step_samples) + num_e_samples, time.time() - st))
                sampled_orders_list = sampler_collection.get_samples(
                    init_orders,
                    num_e_samples,
                    burn_in,
                )
                burn_in = 0

                # the last sampled mutation order from each list
                # use this iteration's sampled mutation orders as initialization for the gibbs samplers next cycle
                init_orders = [sampled_orders[-1].mutation_order for sampled_orders in sampled_orders_list]
                # flatten the list of samples to get all the samples
                e_step_samples += [o for orders in sampled_orders_list for o in orders]

                # Do M-step
                log.info("M STEP, iter %d, time %f" % (run, time.time() - st))

                problem = self.problem_solver_cls(self.feat_generator, e_step_samples, penalty_param)
                theta, pen_exp_log_lik = problem.solve(
                    init_theta=prev_theta,
                    max_iters=self.max_m_iters,
                    num_threads=self.num_threads,
                )
                log.info("Current Theta")
                log.info("\n".join(["%d: %.2g" % (i, theta[i]) for i in range(theta.size) if np.abs(theta[i]) > 1e-5]))
                log.info("penalized log likelihood %f" % pen_exp_log_lik)

                if prev_pen_exp_log_lik is not None:
                    # Get statistics
                    log_lik_vec = problem.calculate_log_lik_ratio_vec(theta, prev_theta)

                    # Calculate lower bound to determine if we need to rerun
                    # Get the confidence interval around the penalized log likelihood (not the log likelihood itself!)
                    ase, lower_bound, _ = get_standard_error_ci_corrected(log_lik_vec, ZSCORE, pen_exp_log_lik - prev_pen_exp_log_lik)
                    print "lower_bound", lower_bound

                    lower_bound_is_negative = (lower_bound < 0)
                    log.info("lower_bound_is_negative %d" % lower_bound_is_negative)
                else:
                    lower_bound_is_negative = False

            if lower_bound_is_negative:
                # if penalized log likelihood is decreasing
                break
            elif prev_pen_exp_log_lik is not None and pen_exp_log_lik - prev_pen_exp_log_lik < diff_thres:
                # if penalized log likelihood is increasing but not by very much
                break

            prev_pen_exp_log_lik = pen_exp_log_lik
            log.info("official pen_exp_log_lik %f" % pen_exp_log_lik)

        return theta
