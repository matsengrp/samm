import sys
import os
import traceback
import pickle
import custom_utils
from custom_utils import CustomCommand
import numpy as np

from common import DEBUG

class ParallelWorker:
    """
    Stores the information for running something in parallel
    These workers can be run throught the ParallelWorkerManager
    """
    def __init__(self, seed):
        """
        @param seed: a seed for for each parallel worker
        """
        raise NotImplementedError()

    def run(self):
        """
        Do not implement this function!
        """
        np.random.seed(self.seed)

        result = None
        try:
            result = self.run_worker()
        except Exception as e:
            print "Exception caught in parallel worker: %s" % e
            traceback.print_exc()
        return result

    def run_worker(self):
        """
        Implement this function!
        Returns whatever value needed from this task
        """
        raise NotImplementedError()

    def  __str__(self):
        """
        @return: string for identifying this worker in an error
        """
        raise NotImplementedError()

class ParallelWorkerManager:
    """
    Runs many ParallelWorkers
    """
    def run(self):
        raise NotImplementedError()

class MultiprocessingManager(ParallelWorkerManager):
    """
    Handles submitting jobs to a multiprocessing pool
    So runs ParallelWorkers using multiple CPUs on the same machine
    """
    def __init__(self, pool, worker_list):
        """
        @param worker_list: List of ParallelWorkers
        """
        self.pool = pool
        self.worker_list = worker_list

    def run(self):
        try:
            # Note that multiprocessing pool will already batch things for you
            results_raw = self.pool.map(run_multiprocessing_worker, self.worker_list)
        except Exception as e:
            print "Error occured when trying to process workers in parallel %s" % e
            # Just do it all one at a time instead
            results_raw = map(run_multiprocessing_worker, self.worker_list)

        results = []
        for i, r in enumerate(results_raw):
            if r is None:
                print "WARNING: multiprocessing worker for this worker failed %s" % self.worker_list[i]
            else:
                results.append(r)
        return results

def run_multiprocessing_worker(worker):
    """
    @param worker: Worker
    Function called on each worker process, used by MultiprocessingManager
    Note: this must be a global function
    """
    return worker.run()

class BatchSubmissionManager(ParallelWorkerManager):
    """
    Handles submitting jobs to a job submission system (e.g. slurm)
    """
    def __init__(self, worker_list, num_approx_batches, worker_folder):
        """
        @param worker_list: List of ParallelWorkers
        @param num_approx_batches: number of batches to make approximately (might be a bit more)
        @param worker_folder: the folder to make all the results from the workers
        """
        self.batch_worker_cmds = []
        self.batched_workers = [] # Tracks the batched workers if something fails
        self.output_files = []
        self.create_batch_worker_cmds(worker_list, num_approx_batches, worker_folder)

    def run(self):
        self.clean_outputs()
        custom_utils.run_cmds(self.batch_worker_cmds)
        res = self.read_batch_worker_results()
        self.clean_outputs()
        return res

    def create_batch_worker_cmds(self, worker_list, num_approx_batches, worker_folder):
        """
        Create commands for submitting to a batch manager
        Pickles the workers as input files to the jobs
        The commands specify the output file names for each job - read these output files
        to retrieve the results from the jobs
        """
        num_workers = len(worker_list)
        num_per_batch = max(num_workers/num_approx_batches, 1)
        for batch_idx, start_idx in enumerate(range(0, num_workers, num_per_batch)):
            batched_workers = worker_list[start_idx:start_idx + num_per_batch]
            self.batched_workers.append(batched_workers)

            # Create the folder for the output from this batch worker
            worker_batch_folder = "%s/batch_%d" % (worker_folder, batch_idx)
            if not os.path.exists(worker_batch_folder):
                os.makedirs(worker_batch_folder)

            # Create the command for this batch worker
            input_file_name = "%s/in.pkl" % worker_batch_folder
            output_file_name = "%s/out.pkl" % worker_batch_folder
            self.output_files.append(output_file_name)
            with open(input_file_name, "w") as cmd_input_file:
                # Pickle the worker as input to the job
                pickle.dump(batched_workers, cmd_input_file)
                cmd_str = "python run_worker.py --input-file %s --output-file %s" % (input_file_name, output_file_name)
                batch_cmd = CustomCommand(
                    cmd_str,
                    outfname=output_file_name,
                    logdir=worker_batch_folder,
                    env=os.environ.copy(),
                )
                self.batch_worker_cmds.append(batch_cmd)

    def read_batch_worker_results(self):
        """
        Read the output (pickle) files from the batched workers
        """
        worker_results = []
        for i, f in enumerate(self.output_files):
            try:
                with open(f, "r") as output_f:
                    res = pickle.load(output_f)
            except Exception as e:
                # Probably the file doesn't exist and the job failed?
                traceback.print_exc()
                print "Rerunning locally -- could not load pickle files %s" % f
                # Now let's try to recover by running the worker
                res = [w.run() for w in self.batched_workers[i]]

            for j, r in enumerate(res):
                if r is None:
                    print "WARNING: multiprocessing worker for this worker failed %s" % self.batched_workers[i][j]
                else:
                    worker_results.append(r)

        return worker_results

    def clean_outputs(self):
        for fname in self.output_files:
            if os.path.exists(fname):
                os.remove(fname)
