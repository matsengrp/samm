import numpy as np
import scipy.misc
import logging as log

from models import ImputedSequenceMutations
from common import *

from sampler_collection import Sampler
from profile_support import profile
from collections import Counter
from mutation_order_gibbs import MutationOrderGibbsSampler

class ChibsSamplerResult:
    def __init__(self, reference_order, log_prob_estimate, log_prob_order, gibbs_samples):
        self.reference_order = reference_order
        self.log_prob_estimate = log_prob_estimate
        self.log_prob_order = log_prob_order
        self.gibbs_samples = gibbs_samples

class MutationOrderChibsSampler(Sampler):
    """
    Perform Chibs method to calculate the marginal likelihood of observing a particular ending sequence
    given the starting sequence
    (Chib 1995. Marginal likelihood from the Gibbs output)

    log p_reforder(end|start,theta) = log p(reference order | start, theta) - log p(reference order | end, start, theta)
    The first log prob term is computed analytically (it's not conditional on the end sequence, so easy to calculate)
    The second log prob term is estimated via samples from gibbs samplers. We split the second lob prob term into multiple
    conditional log probility terms:
    log p(reference order | end, start, theta) = log p(partial order of 2 positions | end, start, theta)
                                                + log p(partial order of 3 positions | end, start, partial order of 2 positions, theta)
                                                ...
                                                + log p(partial order of n-1 positions | end, start, partial order of n-2 positions, theta)
                                                + log p(partial order of n positions | end, start, partial order of n-1 positions, theta)
    Each of these log probability terms is estimated using the empirical distribution of a (conditional) gibbs sampler.
    Note that the last term can be calculated analytically.
    """
    def run(self, init_order, burn_in, num_samples, sampling_rate=0):
        """
        Perform Chibs method to calculate the marginal likelihood of observing a particular ending sequence
        given the starting sequence
        (Chib 1995. Marginal likelihood from the Gibbs output)

        @param init_order: a mutation order to initialize the sampler (list of integers)
        @param burn_in: number of iterations for burn in for the first gibbs sampler
        @param num_samples: number of samples to collect for each gibbs sampler
        """
        for _ in range(self.num_tries):
            # Sample gibbs again but not conditional on any partial ordering
            gibbs_sampler = MutationOrderGibbsSampler(self.theta, self.feature_generator, self.obs_seq_mutation)
            first_sampler_res = gibbs_sampler.run(
                init_order,
                burn_in,
                num_samples,
                sampling_rate=sampling_rate,
            )

            # Choose reference order to be the most commonly seen mutation order
            samples = first_sampler_res.samples
            ctr = Counter([".".join(map(str, s.mutation_order)) for s in samples])
            most_common_order = ctr.most_common(1)[0]
            num_appears = most_common_order[1]
            reference_order = [int(p) for p in most_common_order[0].split(".")]

            # Get estimate of probability of order given starting sequence
            log_prob_order_terms = self._estimate_order_probability(gibbs_sampler, reference_order, samples, num_samples)
            if log_prob_order_terms is None:
                num_samples *= 10
                log.info("Chibs: not enough samples to estimate validation log likelihood -- trying again with %d samples" % num_samples)
                continue

            # Get log probability of reference order (not conditional on ending sequence)
            log_prob_ref_order = gibbs_sampler.get_log_probs(reference_order)

            return ChibsSamplerResult(
                reference_order=reference_order,
                log_prob_estimate=np.sum(log_prob_order_terms),
                log_prob_order=log_prob_ref_order,
                gibbs_samples=first_sampler_res.samples,
            )

    def _estimate_order_probability(self, gibbs_sampler, reference_order, first_samples, num_samples):
        """
        Get estimate of log p(reference order | end, start, theta)
        @param gibbs_sampler: the gibbs sampler for this particular start and ending mutation sequence
        @param reference_order: the mutation order of interest
        @param first_samples: samples drawn from the unconditional gibbs sampler
        @param num_samples: number of samples to draw from each conditional gibbs sampler

        When we split log p(reference order | end, start, theta), we are going to split it from back to front.
        So if the reference order is [10, 3, 8, 2], we estimate the conditional probabilities for
            1. partial ordering [8, 2]
            2. partial ordering [3, 8, 2]
            3. mutation ordering [10, 3, 8, 2]
        """

        if self.num_mutations <= 1:
            # If there are one or fewer mutations the Chibs step is trivial, so output logprob of zero
            return 0.

        log_prob_order_terms = []

        ref_partial_order = reference_order[-1:]
        if self.num_mutations > 2:
            ref_partial_order = reference_order[-2:]
            # The probability of seeing a partial ordering of two positions is estimated from the unconditional gibbs sampler
            num_agrees_normalized = MutationOrderChibsSampler._get_proportion_agrees(first_samples, ref_partial_order)
            if num_agrees_normalized == 0:
                return None

            log_prob_order_terms.append(np.log(num_agrees_normalized))

            for i in range(3, self.num_mutations):
                # Sample gibbs again but with conditional partial order
                sampler_res = gibbs_sampler.run(
                    reference_order,
                    burn_in=0,
                    num_samples=num_samples,
                    sampling_rate=0,
                    conditional_partial_order=ref_partial_order,
                )

                ref_partial_order = reference_order[-i:]
                # Now estimate the probability of seeing the partial ordering of i positions from this conditional gibbs sampler
                num_agrees_normalized = MutationOrderChibsSampler._get_proportion_agrees(sampler_res.samples, ref_partial_order[:2])
                if num_agrees_normalized == 0:
                    return None
                log_prob_order_terms.append(np.log(num_agrees_normalized))

        # Calculate the probability of seeing the full mutation order given a partial ordering of all the other positions
        # This is done analytically. No need for a gibbs sampler.
        # First get the probability of all the mutation orders consistent with this partial ordering.
        _, _, all_log_probs = gibbs_sampler._do_gibbs_step(ref_partial_order, reference_order[0])
        # Normalize across the probability of all consistent mutation orders to get the probability of the full mutation order
        # given the n-1 partial ordering
        log_conditional_prob = all_log_probs[-1] - scipy.misc.logsumexp(all_log_probs)
        log_prob_order_terms.append(log_conditional_prob)

        # The final estimate of the log probability is the sum of all the log conditional probabilities
        return np.sum(log_prob_order_terms)

    @staticmethod
    def _get_proportion_agrees(samples, ref_partial_order):
        """
        Get the proportion of sampled mutation orders that agree with this reference partial order

        @param ref_partial_order: list of two positions in partial mutation order
        """
        assert(len(ref_partial_order) == 2)
        num_agrees = 0
        for s in samples:
            for p in s.mutation_order:
                if p == ref_partial_order[0]:
                    # agrees with partial order
                    num_agrees += 1
                elif p == ref_partial_order[1]:
                    # doesn't agree
                    break
        return float(num_agrees)/len(samples)
