import sys
import traceback
from multiprocessing import Pool
from models import ImputedSequenceMutations

class SamplerCollection:
    """
    A class that will run samplers in parallel.
    A sampler is created for each element in observed_data.
    """
    def __init__(self, observed_data, theta, sampler_cls, feat_generator, num_threads):
        """
        @param observed_data: list of ObservedSequenceMutations objects
        @param theta: numpy vector
        @param sampler_cls: class that inherits from Sampler class
        @param feat_generator: FeatureGenerator
        @param num_threads: number of processes to create when performing gibbs sampling
        """
        self.num_threads = num_threads

        self.samplers = [
            sampler_cls(
                theta,
                feat_generator,
                obs_seq,
            )
            for obs_seq in observed_data
        ]

    def get_samples(self, init_orders_for_iter, num_samples, burn_in_sweeps=0):
        """
        @param init_orders_for_iter: what order to initialize each gibbs sampler
        @param num_samples: number of samples to retrieve from each gibbs sampler
        @param burn_in_sweeps: number of samplers to run initially for burn in
        @returns List of samples from each sampler
        """
        pool = Pool(self.num_threads)
        sampled_orders_list = pool.map(
            _run_sampler,
            [
                SamplerPoolWorker(sampler, init_order, num_samples, burn_in_sweeps)
                for sampler, init_order in zip(self.samplers, init_orders_for_iter)
            ],
        )
        pool.close()

        # If one of the gibbs samplers fails, we should... fail the whole thing for now.
        # We'll deal with this when this becomes a real issue. Maybe we need to try to
        # rerun that particular sampler.
        assert(None not in sampled_orders_list)

        return sampled_orders_list

def _run_sampler(sampler_worker):
    """
    @param sampler_worker: SamplerPoolWorker
    Function called by each worker process in the multiprocessing pool
    Note: this must be a global function
    """
    result = None
    try:
        result = sampler_worker.run()
    except Exception as e:
        print "Exception caught: %s" % e
        traceback.print_exc()
    return result

class SamplerPoolWorker:
    """
    Stores the information for running a sampler
    """
    def __init__(self, sampler, init_order, num_samples, burn_in_sweeps):
        self.sampler = sampler
        self.num_samples = num_samples
        self.burn_in_sweeps = burn_in_sweeps
        self.init_order = init_order

    def run(self):
        samples = self.sampler.run(self.init_order, self.num_samples, self.burn_in_sweeps)
        return samples

class Sampler:
    def __init__(self, theta, feature_generator, obs_seq_mutation):
        """
        @param theta: numpy vector of model parameters
        @param feature_generator: FeatureGenerator
        @param obs_seq_mutation: a ObservedSequenceMutations to know the starting and ending sequence
        """
        self.theta = theta
        self.feature_generator = feature_generator
        self.obs_seq_mutation = obs_seq_mutation

    def run(self, init_order, num_samples, burn_in):
        """
        @param init_order: a mutation order to initialize the sampler (list of integers)
        @param burn_in: number of iterations for burn in
        @param num_samples: number of samples needed
        @return list of ImputedSequenceMutations
        """
        raise NotImplementedError
