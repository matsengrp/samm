"""
ParallelWorker that fits penalized models given training set and penalty parameter.
"""

from parallel_worker import ParallelWorker
from likelihood_evaluator import LikelihoodComparer


class SammWorker(ParallelWorker):
    def __init__(self,
            seed,
            context_model_algo,
            train_set,
            penalty_params,
            max_em_iters,
            val_set_evaluator,
            init_theta,
            reference_pen_param,
            val_set,
            num_val_samples,
            args):
        self.seed = seed
        self.context_model_algo = context_model_algo
        self.train_set = train_set
        self.penalty_params = penalty_params
        self.max_em_iters = max_em_iters
        self.val_set_evaluator = val_set_evaluator
        self.init_theta = init_theta
        self.reference_pen_param = reference_pen_param
        self.val_set = val_set
        self.num_val_samples = num_val_samples
        self.args = args

    def run_worker(self, shared_obj):
        model_results = self.context_model_algo.fit_penalized(
                self.train_set,
                self.penalty_params,
                max_em_iters=self.max_em_iters,
                val_set_evaluator=self.val_set_evaluator,
                init_theta=self.init_theta,
                reference_pen_param=self.reference_pen_param)

        val_set_evaluator = LikelihoodComparer(
            self.val_set,
            self.context_model_algo.feat_generator,
            theta_ref=model_results.penalized_theta,
            num_samples=self.num_val_samples,
            burn_in=self.args.num_val_burnin,
            num_jobs=self.args.num_jobs,
            scratch_dir=self.args.scratch_dir,
        )
        return model_results, val_set_evaluator
