import numpy as np
import scipy.misc
import logging as log

from models import ImputedSequenceMutations
from common import *

from sampler_collection import Sampler
from sampler_collection import SamplerResult
from profile_support import profile

class GibbsStepInfo:
    """
    Store the state of each gibbs sample and intermediate computations
    """
    def __init__(self, order, log_numerators, denominators):
        """
        @param order: a list with positions in the order the mutations happened
        @param log_numerators: the log of the exp(theta * psi) term in the numerator of the likelihood
                            at each mutation step
        @param denominators: the sum of the exp(theta * psi) terms in the denominator of the likelihood
                            at each mutation step
        """
        self.order = order
        self.log_numerators = log_numerators
        self.denominators = denominators

class MutationOrderGibbsSampler(Sampler):
    """
    A class that will do the heavy lifting of Gibbs sampling.
    Returns an order and log probability vector (for tracing)
    """
    def run(self, init_order, burn_in, num_samples):
        traces = []
        if self.num_mutations < 2:
            # If there are zero or one mutations then the same initial order will be returned for
            # every sample
            samples = [init_order] * (burn_in + num_samples)
        else:
            samples = []
            log.info("Gibbs: num mutations %d, seq len %d" % (self.num_mutations, self.obs_seq_mutation.seq_len))

            curr_gibbs_step_info = None
            curr_order = init_order
            for i in range(burn_in + num_samples):
                curr_gibbs_step_info, trace = self._do_gibbs_sweep(curr_order, curr_gibbs_step_info)
                curr_order = curr_gibbs_step_info.order
                samples.append(curr_order)
                traces += trace

        sampled_orders = samples[-num_samples:]
        return SamplerResult(
            [ImputedSequenceMutations(self.obs_seq_mutation, order) for order in sampled_orders],
            traces
        )

    def _do_gibbs_sweep(self, curr_order, gibbs_step_info=None):
        """
        One gibbs sweep is a gibbs sampling step for all the positions
        Returns an order and trace

        @param curr_order: current order in sampling step
        @param gibbs_step_info: GibbsStepInfo with the information from the most recent step.
                                used to minimize recomputation
        """
        trace = []
        # Perform gibbs sweep in a random order
        for position in np.random.permutation(self.mutated_positions):
            # Take out the position we are going to sample order for and get the partial ordering under consideration
            pos_order_idx = curr_order.index(position)
            partial_order = curr_order[0:pos_order_idx] + curr_order[pos_order_idx + 1:]

            gibbs_step_info, log_lik = self._do_gibbs_step(partial_order, position, gibbs_step_info, pos_order_idx)
            curr_order = gibbs_step_info.order
            # Output probabilities for trace
            trace.append(log_lik)
        return gibbs_step_info, trace

    def _do_gibbs_step(self, partial_order, position, gibbs_step_info=None, pos_order_idx=None):
        """
        Performs a single gibbs step
        @param partial_order: the partial order under consideration
        @param position: the position we are trying to determine the full ordering for
        @param gibbs_step_info: if None, compute all features from scratch. otherwise use info from a previous gibbs step
                                to minimize calculations in this step
        @param pos_order_idx: the position's mutation step from the order in `gibbs_step_info`
        """
        # A holder for all the log probs - we need to normalize these later to get our sampled order
        all_log_probs = []

        # Compute probabilities for the orderings under consideration
        # First consider the full ordering with position under consideration mutating last
        order_last = partial_order + [position]
        if gibbs_step_info is None:
            feat_mutation_steps, log_numerators, denominators = self._compute_log_probs_from_scratch(
                order_last,
            )
        else:
            feat_mutation_steps, log_numerators, denominators = self._compute_log_probs_with_reference(
                order_last,
                gibbs_step_info,
                update_step_start=pos_order_idx,
            )

        full_ordering_log_prob = np.sum(log_numerators) - (np.log(denominators)).sum()

        # Add the log probability of the position mutating last
        all_log_probs.append(full_ordering_log_prob)

        # These lists store the history of the log numerators and the denominators
        # The numerators have a possibility of changing three times during order shuffling
        # The denominator have a possibility of changing two times during ordering shuffling
        # We can reconstruct the numerators and the denominators for a particular sampled mutation order
        # from these lists
        log_numerator_hist = [[log_n] for log_n in log_numerators]
        log_numerator_hist[-1].append(log_numerators[-1])
        denominator_hist = [[d] for d in denominators]

        # Now unmutate the string by one mutation step so that we can figure out the features at the positions
        flanked_seq = unmutate_string(
            self.obs_seq_mutation.end_seq_with_flanks,
            unmutate_pos=self.motif_len/2 + position,
            orig_nuc=self.obs_seq_mutation.start_seq[position]
        )
        already_mutated_pos_set = set(partial_order)
        # iterate through the rest of the possible full mutation orders consistent with this partial order
        for idx, i in enumerate(reversed(range(self.num_mutations - 1))):
            possible_full_order = partial_order[:i] + [position] + partial_order[i:]
            shuffled_position = partial_order[i]
            already_mutated_pos_set.remove(shuffled_position)
            # Now unmutate the string so that we can figure out the features at the positions
            # right before the i-th mutation step occured
            flanked_seq = unmutate_string(
                flanked_seq,
                unmutate_pos=self.motif_len/2 + shuffled_position,
                orig_nuc=self.obs_seq_mutation.start_seq[shuffled_position]
            )
            # Now get the features - we only need the feature of the mutating position at the ith step
            # And the feature updates at the time of the `i+1`-th step
            first_mutation_feat, second_feat_mut_step = self.feature_generator.get_shuffled_mutation_steps_delta(
                ImputedSequenceMutations(
                    self.obs_seq_mutation,
                    possible_full_order
                ),
                update_step=i,
                flanked_seq=flanked_seq,
                already_mutated_pos=already_mutated_pos_set,
            )

            # correct the full ordering probability by taking away the old terms
            full_ordering_log_prob += -log_numerators[i] - log_numerators[i + 1] + np.log(denominators[i + 1])

            col_idx_earlier = get_target_col(self.obs_seq_mutation, position) if self.per_target_model else 0
            log_numerators[i] = np.asscalar(self.theta[first_mutation_feat, col_idx_earlier])

            col_idx_later = get_target_col(self.obs_seq_mutation, shuffled_position) if self.per_target_model else 0
            log_numerators[i + 1] = np.asscalar(self.theta[second_feat_mut_step.mutating_pos_feat, col_idx_later])
            denominators[i + 1] = self._get_denom_update(denominators[i], first_mutation_feat, second_feat_mut_step)

            # correct the full ordering probability by adding back the new terms
            full_ordering_log_prob += log_numerators[i] + log_numerators[i + 1] - np.log(denominators[i + 1])

            all_log_probs.append(full_ordering_log_prob)

            # Track the numerator and denominator history
            log_numerator_hist[i].append(log_numerators[i])
            log_numerator_hist[i + 1].append(log_numerators[i + 1])
            denominator_hist[i+1].append(denominators[i + 1])

        # Now sample and reconstruct our decision from the numerator/denominator histories
        gibbs_step_info, log_lik = self._sample_order(
            all_log_probs,
            partial_order,
            position,
            denominator_hist,
            log_numerator_hist,
        )
        return gibbs_step_info, log_lik

    def _sample_order(self, all_log_probs, partial_order, position, denominator_hist, log_numerator_hist):
        """
        Sample mutation order from all the possible full mutation orders
        Reconstruct the intermediate calculations for this sampled order for faster future computations

        @param partial_order: the partial ordering we were originally considering
        @param position: the position we need to determine the order for
        @param denominator_hist: the history of all the denominators for reconstructing the intermediate
                                calculations for the sampled mutation order
        @param log_numerator_hist: the history of all the (log) numerators for reconstructing the intermediate
                                calculations for the sampled mutation order

        @return tuple of GibbsStepInfo and log likelihood of the sampled mutation order
        """
        all_probs = np.exp(all_log_probs)
        sampled_idx = sample_multinomial(all_probs)

        # Now reconstruct our decision
        idx = self.num_mutations - sampled_idx - 1
        sampled_order = partial_order[:idx] + [position] + partial_order[idx:]
        sampled_denominators = (
            [denominator_hist[0][0]]
            + [denominator_hist[i][0] for i in range(1, self.num_mutations - sampled_idx)]
            + [denominator_hist[i][1] for i in range(self.num_mutations - sampled_idx, self.num_mutations)]
        )
        sampled_log_numerators = (
            [log_numerator_hist[i][0] for i in range(self.num_mutations - sampled_idx - 1)]
            + [log_numerator_hist[self.num_mutations - sampled_idx - 1][1]]
            + [log_numerator_hist[i][2] for i in range(self.num_mutations - sampled_idx, self.num_mutations)]
        )

        gibbs_step_sample = GibbsStepInfo(
            sampled_order,
            sampled_log_numerators,
            sampled_denominators,
        )

        return gibbs_step_sample, all_log_probs[sampled_idx]

    def _compute_log_probs_from_scratch(self, curr_order):
        """
        Compute the log likelihood for this full mutation ordering
        Calculates the log likelihood of each mutation step

        @params curr_order: the full mutation ordering under consideration
        @return tuple of
            1. feature mutation steps
            2. the log numerators in the log likelihood of each mutation step
            3. the denominator in the log likelihood of each mutation step
        """
        feat_mutation_steps = self.feature_generator.create_for_mutation_steps(
            ImputedSequenceMutations(
                self.obs_seq_mutation,
                curr_order,
            )
        )

        # Get the components -- numerators and the denominators
        log_numerators = []
        for i, mut_step in enumerate(feat_mutation_steps):
            col_idx = get_target_col(self.obs_seq_mutation, curr_order[i]) if self.per_target_model else 0
            log_numerators.append(np.asscalar(self.theta[mut_step.mutating_pos_feat, col_idx]))

        denominators = [
            (np.exp(self.obs_seq_mutation.feat_matrix_start * self.theta)).sum()
        ]
        prev_feat_mut_step = feat_mutation_steps[0]
        for i, feat_mut_step in enumerate(feat_mutation_steps[1:]):
            new_denom = self._get_denom_update(denominators[i], prev_feat_mut_step.mutating_pos_feat, feat_mut_step)
            prev_feat_mut_step = feat_mut_step
            denominators.append(new_denom)
        return feat_mutation_steps, log_numerators, denominators

    def _compute_log_probs_with_reference(self, curr_order, gibbs_step_base, update_step_start=0):
        """
        Compute probabilities that takes advantage of precomputed information from `gibbs_step_base`

        @params curr_order: the full mutation ordering under consideration
        @param gibbs_step_base: GibbsStepInfo, the reference mutation order and its intermediate computations
        @param update_step_start: which mutations step to start recalculating the log liklihood terms from

        @return similar to _compute_log_probs_from_scratch
        """
        # Update if we have previous computations
        feat_mutation_steps = self.feature_generator.create_remaining_mutation_steps(
            ImputedSequenceMutations(
                self.obs_seq_mutation,
                curr_order,
            ),
            update_step_start=update_step_start,
        )
        # Use gibbs_step_base to update denoms and numerators
        log_numerators = gibbs_step_base.log_numerators[:update_step_start]
        for i, mut_step in enumerate(feat_mutation_steps):
            col_idx = get_target_col(self.obs_seq_mutation, curr_order[update_step_start + i]) if self.per_target_model else 0
            log_numerators.append(np.asscalar(self.theta[mut_step.mutating_pos_feat, col_idx]))

        denominators = gibbs_step_base.denominators[:update_step_start + 1]
        prev_feat_mut_step = feat_mutation_steps[0]
        for i in range(update_step_start, self.num_mutations - 1):
            feat_mut_step = feat_mutation_steps[i - update_step_start + 1]
            new_denom = self._get_denom_update(denominators[i], prev_feat_mut_step.mutating_pos_feat, feat_mut_step)
            prev_feat_mut_step = feat_mut_step
            denominators.append(new_denom)

        return feat_mutation_steps, log_numerators, denominators

    def _get_denom_update(self, old_denominator, prev_feat_idx, feat_mut_step):
        """
        Calculate the denominator of the next mutation step quickly by reusing past computations
        and incorporating the deltas appropriately

        @param old_denominator: the denominator from the previous mutation step
        @param old_log_numerator: the numerator from the previous mutation step
        @param feat_mut_step: the features that differed for this next mutation step
        """
        old_feat_exp_theta_sums = [self.exp_theta_sum[feat_idx] for feat_idx in feat_mut_step.neighbors_feat_old.values()]
        new_feat_exp_theta_sums = [self.exp_theta_sum[feat_idx] for feat_idx in feat_mut_step.neighbors_feat_new.values()]
        return old_denominator - self.exp_theta_sum[prev_feat_idx] - sum(old_feat_exp_theta_sums) + sum(new_feat_exp_theta_sums)
