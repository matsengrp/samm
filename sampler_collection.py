import sys
import numpy as np
import traceback
from multiprocessing import Pool
from models import ImputedSequenceMutations
from parallel_worker import ParallelWorker, BatchSubmissionManager, run_multiprocessing_worker
from common import get_randint, NUM_NUCLEOTIDES
import custom_utils

class SamplerCollection:
    """
    A class that will run samplers in parallel.
    A sampler is created for each element in observed_data.
    """
    def __init__(self, observed_data, theta, sampler_cls, feat_generator, num_jobs, approx):
        """
        @param observed_data: list of ObservedSequenceMutationsFeatures objects
        @param theta: numpy vector
        @param sampler_cls: class that inherits from Sampler class
        @param feat_generator: FeatureGenerator
        @param num_jobs: number of jobs to submit when performing gibbs sampling
        @param approx: level of approximation to use to speed up Gibbs sampling
        """
        self.num_jobs = num_jobs

        self.samplers = [
            sampler_cls(
                theta,
                feat_generator,
                obs_seq,
                approx,
            )
            for obs_seq in observed_data
        ]

    def get_samples(self, init_orders_for_iter, num_samples, burn_in_sweeps=0):
        """
        @param init_orders_for_iter: what order to initialize each gibbs sampler
        @param num_samples: number of samples to retrieve from each gibbs sampler
        @param burn_in_sweeps: number of samplers to run initially for burn in
        @returns List of samples from each sampler (ImputedSequenceMutations) and log probabilities for tracing

        """
        rand_seed = get_randint()
        worker_list = [
            SamplerPoolWorker(rand_seed + i, sampler, init_order, num_samples, burn_in_sweeps)
            for i, (sampler, init_order) in enumerate(zip(self.samplers, init_orders_for_iter))
        ]
        if self.num_jobs > 1:
            batch_manager = BatchSubmissionManager(worker_list, self.num_jobs, "_output/gibbs_workers")
            # TODO: what to do if fails?
            sampled_orders_list = batch_manager.run()
        else:
            sampled_orders_list = map(run_multiprocessing_worker, worker_list)

        return sampled_orders_list

class SamplerPoolWorker(ParallelWorker):
    """
    Stores the information for running a sampler
    """
    def __init__(self, seed, sampler, init_order, num_samples, burn_in_sweeps):
        self.seed = seed
        self.sampler = sampler
        self.num_samples = num_samples
        self.burn_in_sweeps = burn_in_sweeps
        self.init_order = init_order

    def _run(self):
        sampler_res = self.sampler.run(self.init_order, self.num_samples, self.burn_in_sweeps)
        return sampler_res

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
    def __init__(self, theta, feature_generator, obs_seq_mutation, approx="none"):
        """
        @param theta: numpy vector of model parameters
        @param feature_generator: FeatureGenerator
        @param obs_seq_mutation: ObservedSequenceMutationsFeatures
        @param approx: level of approximation to use to speed up Gibbs sampling (none, faster)
        """
        self.theta = theta
        self.per_target_model = self.theta.shape[1] == NUM_NUCLEOTIDES
        self.feature_generator = feature_generator
        self.motif_len = self.feature_generator.motif_len
        self.obs_seq_mutation = obs_seq_mutation
        self.seq_len = obs_seq_mutation.seq_len
        self.mutated_positions = obs_seq_mutation.mutation_pos_dict.keys()
        self.num_mutations = len(self.mutated_positions)
        self.approx = approx

    def run(self, init_order, burn_in, num_samples):
        """
        @param init_order: a mutation order to initialize the sampler (list of integers)
        @param burn_in: number of iterations for burn in
        @param num_samples: number of samples needed
        @return SamplerResult
        """
        raise NotImplementedError
