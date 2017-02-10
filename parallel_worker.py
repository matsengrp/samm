import os
import traceback
import pickle
import utils
from utils import CustomCommand
import numpy as np

class ParallelWorker:
    """
    Stores the information for running something in parallel
    This can be used in two ways:
    1. Submit ParallelWorkers to a job submission system, e.g. slurm
    2. Run ParallelWorkers on multiple CPUs on the same machine
    """
    def __init__(self, seed):
        """
        @param seed: a seed for for each parallel worker
        """
        raise NotImplementedError()

    def run(self):
        np.random.seed(self.seed)
        return self._run()

    def _run(self):
        """
        Returns whatever value needed from this task
        """
        raise NotImplementedError()

class BatchSubmissionManager:
    """
    Handles submitting jobs to a cluster
    """
    def __init__(self, worker_list, num_approx_batches, worker_folder):
        """
        @param worker_list: List of ParallelWorkers
        @param num_approx_batches: number of batches to make approximately (might be a bit more)
        @param worker_folder: the folder to make all the results from the workers
        """
        self.batch_worker_cmds = []
        self.output_files = []
        self.create_batch_worker_cmds(worker_list, num_approx_batches, worker_folder)

    def run(self):
        utils.run_cmds(self.batch_worker_cmds)
        res = self.read_batch_worker_results()
        return res

    def create_batch_worker_cmds(self, worker_list, num_approx_batches, worker_folder):
        """
        Create commands for submitting to a batch manager
        Pickles the workers as input files to the jobs
        The commands specify the output file names for each job - read these output files
        to retrieve the results from the jobs
        """
        num_workers = len(worker_list)
        num_per_batch = num_workers/num_approx_batches
        for batch_idx, start_idx in enumerate(range(0, num_workers, num_per_batch)):
            batched_workers = worker_list[start_idx:start_idx + num_per_batch]

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
        for f in self.output_files:
            with open(f, "r") as output_f:
                res = pickle.load(output_f)
            worker_results += res
        return worker_results

def run_multiprocessing_worker(worker):
    """
    @param worker: Worker
    Function called by each worker process in the multiprocessing pool
    Note: this must be a global function
    """
    np.random.seed(worker.seed)

    result = None
    try:
        result = worker.run()
    except Exception as e:
        print "Exception caught: %s" % e
        traceback.print_exc()
    return result
