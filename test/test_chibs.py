import unittest
import numpy as np
import scipy.misc

from models import ObservedSequenceMutations
from submotif_feature_generator import SubmotifFeatureGenerator
from likelihood_evaluator import LogLikelihoodEvaluator
from mutation_order_gibbs import MutationOrderGibbsSampler

class Chibs_TestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(1)
        cls.motif_len = 3
        cls.burn_in = 10

        cls.feat_gen = SubmotifFeatureGenerator(cls.motif_len)
        cls.theta = np.random.rand(cls.feat_gen.feature_vec_len, 1) * 2

    def test_chibs(self):
        """
        Check that chibs is calculating the correct marginal log likelihood
        """
        # Only 2 mutations
        self._test_chibs_for_obs_seq_mut(
            ObservedSequenceMutations("attcgta", "ataagta", self.motif_len)
        )
        # Many mutations
        self._test_chibs_for_obs_seq_mut(
            ObservedSequenceMutations("attacacgta", "attgggggta", self.motif_len)
        )
        # Only 1 mutation
        self._test_chibs_for_obs_seq_mut(
            ObservedSequenceMutations("attcgta", "attagta", self.motif_len)
        )

    def _test_chibs_for_obs_seq_mut(self, obs_seq_mut):
        obs_seq_m = self.feat_gen.create_base_features(obs_seq_mut)

        gibbs_sampler = MutationOrderGibbsSampler(self.theta, self.feat_gen, obs_seq_m)
        fake_chibs_ll = self.get_log_lik_obs_seq_fake_chibs(
            gibbs_sampler,
            obs_seq_m,
            num_samples=obs_seq_mut.num_mutations * 5000
        )

        val_set_evaluator = LogLikelihoodEvaluator([obs_seq_m], self.feat_gen)
        real_chibs_ll = val_set_evaluator.get_log_lik(self.theta, burn_in=self.burn_in)
        self.assertTrue(np.abs(fake_chibs_ll - real_chibs_ll) < 0.01)

    def get_log_lik_obs_seq_fake_chibs(self, sampler, obs_seq_m, num_samples=1000):
        """
        Get the log likelihood of this ending sequence given the starting sequence
        via a fake Chibs estimation procedure. This method only works if there are
        not a lot of positions mutating. It is useful for comparison against the
        true Chibs implementation (which is much more complicated).

        @param sampler: the MutationOrderGibbsSampler for this observed sequence mutation and theta
        """
        sampler_res = sampler.run(
            obs_seq_m.mutation_pos_dict.keys(),
            self.burn_in,
            num_samples,
        )
        obs_seq_samples = sampler_res.samples

        # p_reforder is the estimate for p(end|start,theta) using a particular reference order. We can calculate it as follows:
        # log p_reforder(end|start,theta) = log p(reference order | start, theta) - log p(reference order | end, start, theta)
        # The first log prob term is computed analytically (it's not conditional on the end sequence, so easy to calculate)
        # The second log prob term is estimated using the empirical distribution of orders from the gibbs sampler
        # We estimate log p(end|start,theta) by taking an average over all orders observed from the gibbs sampler
        # So log p(end|start,theta) = log(mean(p_reforder(end|start,theta)))

        count_dict = {}
        for s in obs_seq_samples:
            mut_order_str = ".".join(map(str, s.mutation_order))
            if mut_order_str not in count_dict:
                count_dict[mut_order_str] = 1
            else:
                count_dict[mut_order_str] += 1

        log_probs = []
        num_sampled_orders = len(count_dict)
        num_samples = len(obs_seq_samples)
        for order_str, order_cnt in count_dict.iteritems():
            reference_order = [int(p) for p in order_str.split(".")]
            log_prob_ref_order = sampler.get_log_probs(reference_order)

            # Count number of times this order appears - this is our estimate of
            # p(reference order | end, start, theta)
            log_prob_order = np.log(float(order_cnt)/num_samples)
            log_probs.append(log_prob_ref_order - log_prob_order)

        log_mean_prob = np.log(np.exp(scipy.misc.logsumexp(log_probs))/num_sampled_orders)
        return log_mean_prob
