import time
import numpy as np
import scipy.misc
import logging as log

from models import ImputedSequenceMutations
from common import *

from sampler_collection import Sampler

class MutationOrderGibbsSampler(Sampler):
    """
    A class that will do the heavy lifting of Gibbs sampling.
    Returns an order and log probability vector (for tracing)
    """
    def run(self, init_order, burn_in, num_samples):
        self.mutated_positions = self.obs_seq_mutation.mutation_pos_dict.keys()
        self.num_mutations = len(self.mutated_positions)

        assert(check_unordered_equal(init_order, self.mutated_positions))

        if self.num_mutations < 2:
            # If there are zero or one mutations then the same initial order will be returned for
            # every sample
            samples = [init_order] * (burn_in + num_samples)
            traces = []
        else:
            samples = []
            traces = []
            log.info("Gibbs: num mutations %d, seq len %d" % (self.num_mutations, self.obs_seq_mutation.seq_len))

            init_dicts, init_seqs, init_log_probs = self._compute_log_probs(init_order)

            curr_order = init_order
            for i in range(burn_in + num_samples):
                curr_order, trace = self._do_gibbs_sweep(curr_order, init_log_probs, init_order, init_dicts, init_seqs)
                samples.append(curr_order)
                traces.append(trace)

        sampled_orders = samples[-num_samples:]

        sample_dict = {}
        sample_dict['sampled_orders'] = [ImputedSequenceMutations(self.obs_seq_mutation, order) for order in sampled_orders]
        sample_dict['trace'] = traces

        return sample_dict

    def _do_gibbs_sweep(self, curr_order, init_log_probs, init_order, init_dicts, init_seqs):
        """
        One gibbs sweep is a gibbs sampling step for all the positions
        Returns an order and trace

        @param curr_order: current order in sampling step
        @param init_log_probs: precalculated probabilities for speeding up computation
        @param init_order: order corresponding to above precalculated probabilities
        @param init_dicts: initial dictionary to pass in instead of initializing on every step
        @param init_seqs: sequences corresponding to above dictionary

        There are three different values self.approx can take here:
        none: no approximations
        faster: uses precalculated probabilities (stored in init_log_probs) from the beginning of
            each Gibbs sweep
        fastest: uses precalculated probabilities (also stored in init_log_probs) from the beginning of
            the entire sampling step
        """

        if self.approx == 'faster':
            # Compute log probs at beginning of sweep
            init_order = curr_order
            _, _, init_log_probs = self._compute_log_probs(init_order, init_dicts, init_seqs)
        elif self.approx == 'none':
            # Compute log probs each iteration
            init_order = None
            init_log_probs = None

        # sample full ordering from conditional prob for this position
        trace = []
        for position in np.random.permutation(self.mutated_positions):
            pos_order_idx = curr_order.index(position)
            partial_order = curr_order[0:pos_order_idx] + curr_order[pos_order_idx + 1:]

            # the probabilities of each full ordering
            full_ordering_log_probs = [0.] * self.num_mutations
            # the orderings under consideration
            # TODO: we can get rid of this variable and just recompute the order
            # It's here right now cause it makes life easy
            full_orderings = [[]] * self.num_mutations

            # first consider the full ordering with position under consideration mutating last
            full_order_last = partial_order + [position]

            feat_vec_dicts, intermediate_seqs, multinomial_sequence = self._compute_log_probs(full_order_last, init_dicts, init_seqs, init_order, init_log_probs)

            full_orderings[0] = full_order_last
            full_ordering_log_probs[0] = np.sum(multinomial_sequence)

            # iterate through the rest of the possible full mutation orders consistent with this partial order
            for idx, i in enumerate(reversed(range(self.num_mutations - 1))):
                if i == self.num_mutations - 2:
                    # If the last mutation, only need to update one feature vec
                    update_steps = [i]
                else:
                    # If not the last mutation, update two feature vecs
                    update_steps = [i, i+1]
                possible_full_order = partial_order[:i] + [position] + partial_order[i:]
                feat_vec_dicts, intermediate_seqs = self.feature_generator.update_for_mutation_steps(
                    ImputedSequenceMutations(
                        self.obs_seq_mutation,
                        possible_full_order
                    ),
                    update_steps=update_steps,
                    base_feat_vec_dicts = feat_vec_dicts,
                    base_intermediate_seqs = intermediate_seqs,
                )

                # calculate multinomial probs - only need to update 2 values (the one where this position mutates and
                # the position that mutates right after it), rest are the same
                full_orderings[idx+1] = possible_full_order

                # multiply the sequence of multinomials to get the probability of the full ordering
                # the product in {eq:full_ordering}
                if self.approx == 'none':
                    multinomial_sequence[i] += \
                        self.theta[feat_vec_dicts[i][possible_full_order[i]]].sum() - \
                        self.theta[feat_vec_dicts[i][full_order_last[i]]].sum()
                    multinomial_sequence[i + 1] = self._get_multinomial_log_prob(feat_vec_dicts[i + 1], possible_full_order[i + 1])
                    full_ordering_log_probs[idx+1] = np.sum(multinomial_sequence)
                else:
                    # Stupid speed up #2:
                    # The change in theta doesn't really change the i+1 element
                    # of our multinomial vector from the previous ith element:
                    full_ordering_log_probs[idx+1] = full_ordering_log_probs[idx] + \
                        self.theta[feat_vec_dicts[i][possible_full_order[i]]].sum() - \
                        self.theta[feat_vec_dicts[i][full_order_last[i]]].sum() + \
                        multinomial_sequence[i] - multinomial_sequence[i + 1]

            # now perform a draw from the multinomial distribution of full orderings
            # the multinomial folows the distribution in {eq:order_conditional_prob}
            sampled_order_idx = sample_multinomial(np.exp(full_ordering_log_probs))
            # update the ordering
            curr_order = full_orderings[sampled_order_idx]
            # Output all log probabilities for trace
            trace.append(full_ordering_log_probs[sampled_order_idx])

        return curr_order, trace

    def _get_multinomial_log_prob(self, feat_vec_dict, numerator_pos):
        """
        a single term in {eq:full_ordering}
        """
        theta_sums = [self.theta[feat_vec].sum() for feat_vec in feat_vec_dict.values()]
        multinomial_prob = self.theta[feat_vec_dict[numerator_pos]].sum() - scipy.misc.logsumexp(theta_sums)
        return multinomial_prob

    def _compute_log_probs(self, curr_order, init_dicts=None, init_seqs=None, init_order=None, init_log_probs=None):
        """
        Driver to compute probabilities with some precompute if statements
        for speeding up computation.

        @params same as in _do_gibbs_sweep
        """
        if init_dicts is None and init_seqs is None:
            # Compute dictionaries if we haven't yet
            feat_vec_dicts, intermediate_seqs = self.feature_generator.create_for_mutation_steps(
                ImputedSequenceMutations(
                    self.obs_seq_mutation,
                    curr_order,
                )
            )
        else:
            # Update dictionaries if they've been computed already
            feat_vec_dicts, intermediate_seqs = self.feature_generator.update_for_mutation_steps(
                ImputedSequenceMutations(
                    self.obs_seq_mutation,
                    curr_order,
                ),
                update_steps=range(self.num_mutations-1),
                base_feat_vec_dicts = init_dicts,
                base_intermediate_seqs = init_seqs,
            )

        if init_order is None and init_log_probs is None:
            # Compute probabilities if they haven't been already
            log_probs = [
                self._get_multinomial_log_prob(feat_vec_dict_step, curr_mutate_pos)
                for curr_mutate_pos, feat_vec_dict_step in zip(curr_order, feat_vec_dicts)
            ]
        else:
            # Otherwise just reorder precomputed probabilities
            log_probs = [init_log_probs[init_order.index(item)] for item in curr_order]

        return feat_vec_dicts, intermediate_seqs, log_probs

class MutationOrderGibbsSamplerMultiTarget(MutationOrderGibbsSampler):
    """
    Deals with different theta vector for each target nucleotide
    """
    def _get_multinomial_prob(self, feat_vec_dict, numerator_pos):
        """
        a single term in {eq:full_ordering}
        """
        # Calculate the components in the risk group (risk group is all positions
        # that have not mutated yet)
        theta_sums = [
            self.theta[feat_vec, i].sum() for pos, feat_vec in feat_vec_dict.iteritems() for i in range(NUM_NUCLEOTIDES)
        ]

        numerator_target_nucleotide = self.obs_seq_mutation.mutation_pos_dict[numerator_pos]
        multinomial_prob = np.exp(
            self.theta[
                feat_vec_dict[numerator_pos], # motif idx
                NUCLEOTIDE_DICT[numerator_target_nucleotide] # target nucleotide
            ].sum() - scipy.misc.logsumexp(theta_sums)
        )
        return multinomial_prob
