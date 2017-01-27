import traceback

class ParallelWorker:
    """
    Stores the information for running something in parallel
    """
    def __init__(self):
        raise NotImplementedError()

    def run(self):
        """
        Returns whatever value needed from this task
        """
        raise NotImplementedError()


def run_parallel_worker(worker):
    """
    @param worker: Worker
    Function called by each worker process in the multiprocessing pool
    Note: this must be a global function
    """
    result = None
    try:
        result = worker.run()
    except Exception as e:
        print "Exception caught: %s" % e
        traceback.print_exc()
    return result
