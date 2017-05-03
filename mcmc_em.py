import time
import numpy as np
import logging as log
import pickle

from models import *
from common import *
from sampler_collection import SamplerCollection
from profile_support import profile
from confidence_interval_maker import ConfidenceIntervalMaker

class MCMC_EM:
    def __init__(self, train_data, val_data, feat_generator, sampler_cls, problem_solver_cls, possible_theta_mask, zero_theta_mask, base_num_e_samples=10, max_m_iters=500, num_jobs=1, scratch_dir='_output', pool=None):
        """
        @param train_data, val_data: lists of ObservedSequenceMutationsFeatures (start and end sequences, plus base feature info)
        @param feat_generator: an instance of a FeatureGenerator
        @param sampler_cls: a Sampler class
        @param problem_solver_cls: SurvivalProblem class
        @param base_num_e_samples: number of E-step samples to draw initially
        @param max_m_iters: maximum number of iterations for the M-step
        @param num_jobs: number of jobs to submit for E-step
        """
        self.train_data = train_data
        self.val_data = val_data
        self.feat_generator = feat_generator
        self.motif_list = self.feat_generator.motif_list
        self.base_num_e_samples = base_num_e_samples
        self.max_m_iters = max_m_iters
        self.sampler_cls = sampler_cls
        self.problem_solver_cls = problem_solver_cls
        self.num_jobs = num_jobs
        self.pool = pool
        self.possible_theta_mask = possible_theta_mask
        self.zero_theta_mask = zero_theta_mask
        self.scratch_dir = scratch_dir
        self.per_target_model = possible_theta_mask.shape[1] == NUM_NUCLEOTIDES + 1

    def run(self, theta, penalty_params=[1], fuse_windows=[], fuse_center_only=False, max_em_iters=10, burn_in=1, diff_thres=1e-6, max_e_samples=20, train_and_val=False, intermed_file_prefix="", get_hessian=False):
        """
        @param theta: initial value for theta in MCMC-EM
        @param penalty_params: the coefficient(s) for the penalty function
        @param max_em_iters: the maximum number of iterations of MCMC-EM
        @param burn_in: number of burn in iterations
        @param diff_thres: if the change in the objective function changes no more than `diff_thres`, stop MCMC-EM
        @param max_e_samples: maximum number of e-samples to grab per observed sequence
        @param train_and_val: whether to train on both train and validation data
        """
        st = time.time()
        observed_data = self.train_data + self.val_data if train_and_val else self.train_data
        num_data = len(observed_data)
        # stores the initialization for the gibbs samplers for the next iteration's e-step
        init_orders = [obs_seq.mutation_pos_dict.keys() for obs_seq in observed_data]
        all_traces = []
        # burn in only at the very beginning
        for run in range(max_em_iters):
            prev_theta = theta
            num_e_samples = self.base_num_e_samples

            sampler_collection = SamplerCollection(
                observed_data,
                prev_theta,
                self.sampler_cls,
                self.feat_generator,
                self.num_jobs,
                self.scratch_dir,
            )

            e_step_samples = []
            e_step_labels = []
            lower_bound_is_negative = True
            while len(e_step_samples)/num_data < max_e_samples and lower_bound_is_negative:
                ## Keep grabbing samples until it is highly likely we have increased the penalized log likelihood

                # do E-step
                log.info("E STEP, iter %d, num samples %d, time %f" % (run, len(e_step_samples)/num_data + num_e_samples, time.time() - st))
                sampler_results = sampler_collection.get_samples(
                    init_orders,
                    num_e_samples,
                    burn_in,
                )
                # Don't use burn-in from now on
                burn_in = 0
                all_traces.append([res.trace for res in sampler_results])
                sampled_orders_list = [res.samples for res in sampler_results]

                # the last sampled mutation order from each list
                # use this iteration's sampled mutation orders as initialization for the gibbs samplers next cycle
                init_orders = [sampled_orders[-1].mutation_order for sampled_orders in sampled_orders_list]
                # flatten the list of samples to get all the samples
                e_step_samples += [o for orders in sampled_orders_list for o in orders]
                e_step_labels += [i for i, orders in enumerate(sampled_orders_list) for o in orders]

                # Do M-step
                log.info("M STEP, iter %d, time %f" % (run, time.time() - st))

                problem = self.problem_solver_cls(
                    self.feat_generator,
                    e_step_samples,
                    e_step_labels,
                    penalty_params,
                    self.per_target_model,
                    possible_theta_mask=self.possible_theta_mask,
                    zero_theta_mask=self.zero_theta_mask,
                    fuse_windows=fuse_windows,
                    fuse_center_only=fuse_center_only,
                    pool=self.pool,
                )

                theta, pen_exp_log_lik, lower_bound = problem.solve(
                    init_theta=prev_theta,
                    max_iters=self.max_m_iters,
                )

                num_nonzero = get_num_nonzero(theta)
                num_unique = get_num_unique_theta(theta)
                log.info("Current Theta, num_nonzero %d, unique %d" % (num_nonzero, num_unique))
                log.info(
                    get_nonzero_theta_print_lines(theta, self.motif_list, self.feat_generator.mutating_pos_list, self.feat_generator.motif_len)
                )
                log.info("penalized log likelihood %f" % pen_exp_log_lik)
                lower_bound_is_negative = (lower_bound < 0)
                log.info("lower_bound_is_negative %d, lower_bound %f" % (lower_bound_is_negative, lower_bound))

                if num_nonzero == 0:
                    # The whole theta is zero - just stop and consider a different penalty parameter
                    break


            # Save the e-step samples if we want to analyze later on
            e_sample_file_name = "%s%d.pkl" % (intermed_file_prefix, run)
            log.info("Pickling E-step samples %s" % e_sample_file_name)
            with open(e_sample_file_name, "w") as f:
                pickle.dump(e_step_samples, f)

            if lower_bound_is_negative or lower_bound < diff_thres:
                # if penalized log likelihood is decreasing - gradient descent totally failed in this case
                break
            log.info("step final pen_exp_log_lik %f" % pen_exp_log_lik)

        if get_hessian:
            ci_maker = ConfidenceIntervalMaker(self.feat_generator.motif_list, self.per_target_model, self.possible_theta_mask, self.zero_theta_mask)
            theta_standard_error = ci_maker.run(theta, e_step_samples, problem)
        else:
            theta_standard_error = None
        return theta, theta_standard_error, all_traces
