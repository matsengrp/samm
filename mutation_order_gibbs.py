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

            gibbs_step_info, log_lik = self._do_gibbs_step(partial_order, position, pos_order_idx, gibbs_step_info)
            curr_order = gibbs_step_info.order
            # Output probabilities for trace
            trace.append(log_lik)
        return gibbs_step_info, trace

    def _do_gibbs_step(self, partial_order, position, pos_order_idx, gibbs_step_info=None):
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

            log_numerators[i] = np.asscalar(self.theta[first_mutation_feat])
            log_numerators[i + 1] = np.asscalar(self.theta[second_feat_mut_step.mutating_pos_feat])
            denominators[i + 1] = self._get_denom_update(denominators[i], log_numerators[i], second_feat_mut_step)

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

    def _update_log_prob_from_shuffle(self, i, old_log_probs, order, prev_order, feature_vec_theta_sums, prev_feature_vec_theta_sums):
        """
        Calculate multinomial probs - only need to update 2 values (the one where this position mutates and
        the position that mutates right after it), rest are the same
        @param i: the index of the positions in the `order` and `prev_order` that are being shuffled in the mutation order
        @param old_log_probs: the previous log probabilities - most of these are correct. only corrections need to occur in i and i + 1
        @param order: the order after the shuffle
        @param prev_order: the order before the shuffle
        @param feature_vec_theta_sums: the theta sum values for each position after the shuffle
        @param prev_feature_vec_theta_sums: the theta sum values for each position before the shuffle
        """
        log_probs = np.copy(old_log_probs)
        pos_mutate_earlier = order[i]
        pos_mutate_later = prev_order[i]

        # Term at `i` is easy to update: just change out the numerator from the multinomial (but log it)
        col_idx_earlier = self._get_target_col(pos_mutate_earlier) if self.per_target_model else 0
        col_idx_later = self._get_target_col(pos_mutate_later) if self.per_target_model else 0
        log_probs[i] += feature_vec_theta_sums[i][pos_mutate_earlier][col_idx_earlier] - feature_vec_theta_sums[i][pos_mutate_later][col_idx_later]

        # Term at `i + 1` is complicated: we need to change out the numerator and some terms in the denominator
        # Remember that the denominator is the sum of exp thetas - some of the thetas are have the wrong value if the mutation order is shuffled
        # We only chose the theta value that are potentially wrong.
        # We are making the assumption that the features only change for positions close to the positions that got shuffled.
        changed_feature_positions = set(
            range(max(pos_mutate_earlier - self.motif_len/2, 0), min(pos_mutate_earlier + self.motif_len/2 + 1, self.seq_len))
            + range(max(pos_mutate_later - self.motif_len/2, 0), min(pos_mutate_later + self.motif_len/2 + 1, self.seq_len))
        ) - set(order[:i])

        old_sumexp = np.exp(prev_feature_vec_theta_sums[i + 1][pos_mutate_earlier][col_idx_earlier] - log_probs[i + 1])

        old_sumexp_terms = np.exp([
            prev_feature_vec_theta_sums[i + 1][p] for p in changed_feature_positions if p != pos_mutate_later
        ])
        new_sumexp_terms = np.exp([feature_vec_theta_sums[i + 1][p] for p in changed_feature_positions if p != pos_mutate_earlier])

        log_probs[i + 1] = feature_vec_theta_sums[i + 1][pos_mutate_later][col_idx_later] - np.log(
            old_sumexp + new_sumexp_terms.sum() - old_sumexp_terms.sum()
        )

        return log_probs

    def _compute_log_probs_from_scratch(self, curr_order):
        feat_mutation_steps = self.feature_generator.create_for_mutation_steps(
            ImputedSequenceMutations(
                self.obs_seq_mutation,
                curr_order,
            )
        )

        # Get the components -- numerators and the denomiators
        log_numerators = [np.asscalar(self.theta[mut_step.mutating_pos_feat]) for mut_step in feat_mutation_steps]
        denominators = [
            (np.exp(self.obs_seq_mutation.feat_matrix_start * self.theta)).sum()
        ]
        for i, feat_mut_step in enumerate(feat_mutation_steps[1:]):
            new_denom = self._get_denom_update(denominators[i], log_numerators[i], feat_mut_step)
            denominators.append(new_denom)

        return feat_mutation_steps, log_numerators, denominators

    def _compute_log_probs_with_reference(self, curr_order, gibbs_step_base=None, update_step_start=0):
        """
        Driver to compute probabilities with some precompute if statements
        for speeding up computation.

        @params same as in _do_gibbs_sweep
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
        log_numerators = (
            gibbs_step_base.log_numerators[:update_step_start]
            + [np.asscalar(self.theta[mut_step.mutating_pos_feat]) for mut_step in feat_mutation_steps]
        )
        denominators = gibbs_step_base.denominators[:update_step_start + 1]
        for i in range(update_step_start, self.num_mutations - 1):
            feat_mut_step = feat_mutation_steps[i - update_step_start + 1]
            new_denom = self._get_denom_update(denominators[i], log_numerators[i], feat_mut_step)
            denominators.append(new_denom)

        # f, n, d = self._compute_log_probs_from_scratch(curr_order)
        #
        # assert(np.allclose(n, log_numerators))
        # assert(np.allclose(d, denominators))

        return feat_mutation_steps, log_numerators, denominators

    def _get_denom_update(self, old_denominator, old_log_numerator, feat_mut_step):
        old_feat_theta_sums = [self.theta[feat_idx] for feat_idx in feat_mut_step.neighbors_feat_old.values()]
        new_feat_theta_sums = [self.theta[feat_idx] for feat_idx in feat_mut_step.neighbors_feat_new.values()]
        return old_denominator - np.exp(old_log_numerator) - (np.exp(old_feat_theta_sums)).sum() + (np.exp(new_feat_theta_sums)).sum()

    def _get_multinomial_log_prob(self, numerator_pos, feature_vec_theta_sum):
        """
        This is for the case where theta is 1-dimensional!
        a single term in {eq:full_ordering}
        """
        # TODO: TAKE THIS OUT - this can get slow! we need to use previous computations
        theta_sums = feature_vec_theta_sum.values()
        col_idx = self._get_target_col(numerator_pos) if self.per_target_model else 0
        multinomial_prob = feature_vec_theta_sum[numerator_pos][col_idx] - scipy.misc.logsumexp(theta_sums)
        return multinomial_prob

    def _get_target_col(self, mutation_pos):
        """
        ONLY call this if self.per_target_model is True
        Determine which column to read out corresponding to the target nucleotide
        """
        # TODO: switch dna sequences to numerical
        target_nucleotide = self.obs_seq_mutation.mutation_pos_dict[mutation_pos]
        return NUCLEOTIDE_DICT[target_nucleotide]
