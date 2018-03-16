import sys
import numpy as np
import traceback
import os.path
from models import ImputedSequenceMutations
from parallel_worker import ParallelWorker
from parallel_worker import BatchSubmissionManager
from parallel_worker import MultiprocessingManager
from common import get_randint, NUM_NUCLEOTIDES
import custom_utils

class SamplerCollection:
    """
    A class that will run samplers in parallel.
    A sampler is created for each element in observed_data.
    """
    def __init__(self, observed_data, theta, sampler_cls, feat_generator, num_jobs=None, scratch_dir=None, pool=None, num_tries=5, get_residuals=False):
        """
        There are two choices for running a sampler collection: Batch submission and multithreading.
        If num_jobs and scratch_dir are specified, then we perform batch submission.
        If both are provided, we perform multithreading

        @param observed_data: list of ObservedSequenceMutationsFeatures objects
        @param theta: numpy vector
        @param sampler_cls: class that inherits from Sampler class
        @param feat_generator: FeatureGenerator
        @param num_jobs: number of jobs to submit when performing gibbs sampling
        @param scratch_dir: a tmp directory to write files in for the batch submission manager
        @param pool: multiprocessing pool previously initialized before model fitting
        @param num_tries: number of tries for Chibs sampler
        """
        self.num_jobs = num_jobs
        self.scratch_dir = scratch_dir
        self.pool = pool

        self.sampler_cls = sampler_cls
        self.theta = theta
        self.feat_generator = feat_generator
        self.observed_data = observed_data
        self.num_tries = num_tries
        self.get_residuals = get_residuals

    def get_samples(self, init_orders_for_iter, num_samples, burn_in_sweeps=0, sampling_rate=1):
        """
        @param init_orders_for_iter: what order to initialize each gibbs sampler
        @param num_samples: number of samples to retrieve from the sampler
        @param burn_in_sweeps: number of samplers to run initially for burn in
        @param sampling_rate: (non-negative integer)
            if 0, then get all the samples in a gibbs sweep
            if K for K > 0, get 1 sample per K gibbs sweeps
        @param conditional_partial_order: list of position for a partial mutation order. if non-empty, then
                                            condition on this conditional_partial_order
        @returns List of samples from each sampler (ImputedSequenceMutations) and log probabilities for tracing
        """
        rand_seed = get_randint()
        shared_obj = SamplerPoolWorkerShared(self.sampler_cls, self.theta, self.feat_generator, num_samples, burn_in_sweeps, sampling_rate, self.num_tries, self.get_residuals)
        worker_list = [
            SamplerPoolWorker(rand_seed + i, obs_data, init_order)
            for i, (obs_data, init_order) in enumerate(zip(self.observed_data, init_orders_for_iter))
        ]
        if self.num_jobs is not None and self.num_jobs > 1:
            batch_manager = BatchSubmissionManager(worker_list, shared_obj, self.num_jobs, os.path.join(self.scratch_dir, "gibbs_workers"))
            sampled_orders_list = batch_manager.run()
        elif self.pool is not None:
            proc_manager = MultiprocessingManager(self.pool, worker_list)
            sampled_orders_list = proc_manager.run()
        else:
            sampled_orders_list = [worker.run(shared_obj) for worker in worker_list]

        return sampled_orders_list

class SamplerPoolWorkerShared:
    def __init__(self, sampler_cls, theta, feat_generator, num_samples, burn_in_sweeps, sampling_rate, num_tries, get_residuals):
        self.sampler_cls = sampler_cls
        self.theta = theta
        self.feat_generator = feat_generator
        self.num_samples = num_samples
        self.burn_in_sweeps = burn_in_sweeps
        self.sampling_rate = sampling_rate
        self.num_tries = num_tries
        self.get_residuals = get_residuals

class SamplerPoolWorker(ParallelWorker):
    """
    Stores the information for running a sampler
    """
    def __init__(self, seed, obs_seq, init_order):
        self.seed = seed
        self.obs_seq = obs_seq
        self.init_order = init_order

    def run_worker(self, shared_obj):
        sampler = shared_obj.sampler_cls(
            shared_obj.theta,
            shared_obj.feat_generator,
            self.obs_seq,
            shared_obj.num_tries,
            shared_obj.get_residuals,
        )
        sampler_res = sampler.run(self.init_order, shared_obj.burn_in_sweeps, shared_obj.num_samples, shared_obj.sampling_rate)
        return sampler_res

    def __str__(self):
        return "SamplerPoolWorker %s" % self.obs_seq

class Sampler:
    def __init__(self, theta, feature_generator, obs_seq_mutation, num_tries=5, get_residuals=False):
        """
        @param theta: numpy vector of model parameters
        @param feature_generator: FeatureGenerator
        @param obs_seq_mutation: ObservedSequenceMutationsFeatures
        @param num_tries: number of tries for Chibs sampler
        """
        self.theta = theta
        self.exp_theta = np.exp(theta)
        self.per_target_model = self.theta.shape[1] == NUM_NUCLEOTIDES + 1

        if len(feature_generator.motif_lens) == 1:
            self.exp_theta_num_cols = 1 if not self.per_target_model else NUM_NUCLEOTIDES
            if not self.per_target_model:
                self.exp_theta_sum = np.exp(theta).sum(axis=1)
            else:
                theta_summed = theta[:,0,None] + theta[:,1:]
                self.exp_theta_sum = np.exp(theta_summed).sum(axis=1)

        self.feature_generator = feature_generator
        self.motif_len = self.feature_generator.motif_len
        self.obs_seq_mutation = obs_seq_mutation
        self.seq_len = obs_seq_mutation.seq_len
        self.mutated_positions = obs_seq_mutation.mutation_pos_dict.keys()
        self.num_mutations = len(self.mutated_positions)

        self.num_tries = num_tries
        self.get_residuals = get_residuals
