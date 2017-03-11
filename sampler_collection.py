import sys
import numpy as np
import traceback
import os.path
from multiprocessing import Pool
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
    def __init__(self, observed_data, theta, sampler_cls, feat_generator, num_jobs=None, scratch_dir=None, num_threads=None):
        """
        There are two choices for running a sampler collection: Batch submission and multithreading.
        If num_jobs and scratch_dir are specified, then we perform batch submission.
        If num_threads is specified, then we perform multithreading
        If both are provided, we perform multithreading

        @param observed_data: list of ObservedSequenceMutationsFeatures objects
        @param theta: numpy vector
        @param sampler_cls: class that inherits from Sampler class
        @param feat_generator: FeatureGenerator
        @param num_jobs: number of jobs to submit when performing gibbs sampling
        @param scratch_dir: a tmp directory to write files in for the batch submission manager
        @param num_threads: number of threads to run to perform gibbs sampling
        """
        self.num_jobs = num_jobs
        self.scratch_dir = scratch_dir

        self.num_threads = num_threads

        self.samplers = [
            sampler_cls(
                theta,
                feat_generator,
                obs_seq,
            )
            for obs_seq in observed_data
        ]

    def get_samples(self, init_orders_for_iter, num_samples, burn_in_sweeps=0, get_full_sweep=False):
        """
        @param init_orders_for_iter: what order to initialize each gibbs sampler
        @param num_samples: number of samples to retrieve from each gibbs sampler
        @param burn_in_sweeps: number of samplers to run initially for burn in
        @param get_full_sweep: whether to return all the gibbs samples (so all steps, not just each sweep)
        @returns List of samples from each sampler (ImputedSequenceMutations) and log probabilities for tracing
        """
        rand_seed = get_randint()

        worker_list = [
            SamplerPoolWorker(rand_seed + i, sampler, init_order, num_samples, burn_in_sweeps, get_full_sweep)
            for i, (sampler, init_order) in enumerate(zip(self.samplers, init_orders_for_iter))
        ]
        if self.num_jobs is not None and self.num_jobs > 1:
            batch_manager = BatchSubmissionManager(worker_list, self.num_jobs, os.path.join(self.scratch_dir, "gibbs_workers"))
            sampled_orders_list = batch_manager.run()
        elif self.num_threads is not None and self.num_threads > 1:
            self.pool = Pool(self.num_threads)
            proc_manager = MultiprocessingManager(self.pool, worker_list)
            sampled_orders_list = proc_manager.run()
            self.pool.close()
            self.pool.join()
        else:
            sampled_orders_list = [worker.run() for worker in worker_list]

        return sampled_orders_list

class SamplerPoolWorker(ParallelWorker):
    """
    Stores the information for running a sampler
    """
    def __init__(self, seed, sampler, init_order, num_samples, burn_in_sweeps, get_full_sweep):
        self.seed = seed
        self.sampler = sampler
        self.num_samples = num_samples
        self.burn_in_sweeps = burn_in_sweeps
        self.init_order = init_order
        self.get_full_sweep = get_full_sweep

    def run_worker(self):
        sampler_res = self.sampler.run(self.init_order, self.burn_in_sweeps, self.num_samples, self.get_full_sweep)
        return sampler_res

    def __str__(self):
        return "SamplerPoolWorker %s" % self.sampler.obs_seq_mutation

class SamplerResult:
    def __init__(self, samples, trace):
        """
        class returned by Sampler after a run

        @param samples: list of ImputedSequenceMutations
        @param trace: list of things to plot for trace diagnostics
        """
        self.samples = samples
        self.trace = trace

class Sampler:
    def __init__(self, theta, feature_generator, obs_seq_mutation):
        """
        @param theta: numpy vector of model parameters
        @param feature_generator: FeatureGenerator
        @param obs_seq_mutation: ObservedSequenceMutationsFeatures
        """
        self.theta = theta
        self.exp_theta = np.exp(theta)
        self.exp_theta_sum = np.exp(theta).sum(axis=1)
        self.per_target_model = self.theta.shape[1] == NUM_NUCLEOTIDES
        self.feature_generator = feature_generator
        self.motif_len = self.feature_generator.motif_len
        self.obs_seq_mutation = obs_seq_mutation
        self.seq_len = obs_seq_mutation.seq_len
        self.mutated_positions = obs_seq_mutation.mutation_pos_dict.keys()
        self.num_mutations = len(self.mutated_positions)

    def run(self, init_order, burn_in, num_samples):
        """
        @param init_order: a mutation order to initialize the sampler (list of integers)
        @param burn_in: number of iterations for burn in
        @param num_samples: number of samples needed
        @return SamplerResult
        """
        raise NotImplementedError
