import copy
import numpy as np
import scipy.misc
import logging as log

from models import ImputedSequenceMutations
from common import *
from decorators import measure_time

from sampler_collection import Sampler
from sampler_collection import SamplerResult

class GibbsStepInfo:
    def __init__(self, order, feat_vec_dicts, intermediate_seqs, log_probs):
        self.order = order
        self.feat_vec_dicts = feat_vec_dicts
        self.intermediate_seqs = intermediate_seqs
        self.log_probs = log_probs

class FullMutationOrderGibbsStepOptions:
    """
    Object for storing information about all the options possible for this gibbs step.
    So this stores all the info related to each possible full mutation order
    that are consistent with the partial ordering
    """
    def __init__(self):
        self.full_ordering_log_probs = []
        self.log_probs_list = []
        self.order_list = []
        self.feat_vec_dicts_list = []
        self.intermediate_seqs_list = []

    def append(self, order, feat_vec_dicts, intermediate_seqs, log_probs):
        """
        Add this new full ordering and its info to the list
        """
        self.order_list.append(order)
        self.feat_vec_dicts_list.append(feat_vec_dicts)
        self.intermediate_seqs_list.append(intermediate_seqs)
        self.log_probs_list.append(list(log_probs))
        self.full_ordering_log_probs.append(
            np.sum(log_probs)
        )

    def sample(self):
        """
        Sample the orders from the current list where full_ordering_log_probs are the weights.
        @returns GibbsStepInfo with the sampled order
        """
        sampled_idx = sample_multinomial(np.exp(self.full_ordering_log_probs))
        gibbs_step_info = GibbsStepInfo(
            self.order_list[sampled_idx],
            self.feat_vec_dicts_list[sampled_idx],
            self.intermediate_seqs_list[sampled_idx],
            self.log_probs_list[sampled_idx],
        )

        sampled_order_probs = self.full_ordering_log_probs[sampled_idx]
        return gibbs_step_info, sampled_order_probs

class MutationOrderGibbsSampler(Sampler):
    """
    A class that will do the heavy lifting of Gibbs sampling.
    Returns an order and log probability vector (for tracing)
    """
    @measure_time
    def run(self, init_order, burn_in, num_samples):
        self.mutated_positions = self.obs_seq_mutation.mutation_pos_dict.keys()
        self.num_mutations = len(self.mutated_positions)

        assert(check_unordered_equal(init_order, self.mutated_positions))

        traces = []
        if self.num_mutations < 2:
            # If there are zero or one mutations then the same initial order will be returned for
            # every sample
            samples = [init_order] * (burn_in + num_samples)
        else:
            samples = []
            log.info("Gibbs: num mutations %d, seq len %d" % (self.num_mutations, self.obs_seq_mutation.seq_len))

            init_dicts, init_seqs, init_log_probs = self._compute_log_probs(init_order)
            curr_gibbs_step_info = GibbsStepInfo(init_order, init_dicts, init_seqs, init_log_probs)
            for i in range(burn_in + num_samples):
                curr_gibbs_step_info, trace = self._do_gibbs_sweep(curr_gibbs_step_info)
                samples.append(curr_gibbs_step_info.order)
                traces += trace

        sampled_orders = samples[-num_samples:]
        return SamplerResult(
            [ImputedSequenceMutations(self.obs_seq_mutation, order) for order in sampled_orders],
            traces
        )

    @measure_time
    def _do_gibbs_sweep(self, gibbs_step_info):
        """
        One gibbs sweep is a gibbs sampling step for all the positions
        Returns an order and trace

        @param curr_order: current order in sampling step
        @param gibbs_step_info: GibbsStepInfo with the information from the most recent step - used to minimize recomputation

        There are three different values self.approx can take here:
        none: no approximations
        faster: uses precalculated probabilities (stored in init_log_probs) from the beginning of
            each Gibbs sweep
        """
        # sample full ordering from conditional prob for this position
        trace = []
        for position in np.random.permutation(self.mutated_positions):
            # Take out the position we are going to sample order for and get the partial ordering under consideration
            pos_order_idx = gibbs_step_info.order.index(position)
            partial_order = gibbs_step_info.order[0:pos_order_idx] + gibbs_step_info.order[pos_order_idx + 1:]

            # Compute probabilities for the orderings under consideration
            # First consider the full ordering with position under consideration mutating last
            full_order_last = partial_order + [position]
            feat_vec_dicts, intermediate_seqs, log_probs = self._compute_log_probs(
                full_order_last,
                gibbs_step_info,
                update_positions=range(pos_order_idx, self.num_mutations),
            )

            gibbs_step_options = FullMutationOrderGibbsStepOptions()
            gibbs_step_options.append(
                full_order_last,
                feat_vec_dicts,
                intermediate_seqs,
                log_probs,
            )

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

                # multiply the sequence of multinomials to get the probability of the full ordering
                # the product in {eq:full_ordering}
                if self.approx == 'none':
                    # calculate multinomial probs - only need to update 2 values (the one where this position mutates and
                    # the position that mutates right after it), rest are the same
                    log_probs[i] += \
                        self.theta[feat_vec_dicts[i][possible_full_order[i]]].sum() - \
                        self.theta[feat_vec_dicts[i][full_order_last[i]]].sum()
                    log_probs[i + 1] = self._get_multinomial_log_prob(feat_vec_dicts[i + 1], possible_full_order[i + 1])
                else:
                    # Stupid speed up #2:
                    # The change in theta doesn't really change the i+1 element
                    # of our multinomial vector from the previous ith element:
                    full_ordering_log_probs[idx+1] = full_ordering_log_probs[idx] + \
                        self.theta[feat_vec_dicts[i][possible_full_order[i]]].sum() - \
                        self.theta[feat_vec_dicts[i][full_order_last[i]]].sum() + \
                        log_probs[i] - log_probs[i + 1]

                gibbs_step_options.append(
                    possible_full_order,
                    feat_vec_dicts,
                    intermediate_seqs,
                    log_probs,
                )

            # now perform a draw from the multinomial distribution of full orderings
            # the multinomial follows the distribution in {eq:order_conditional_prob}
            gibbs_step_info, curr_order_probs = gibbs_step_options.sample()

            # Output all log probabilities for trace
            trace.append(curr_order_probs)

        return gibbs_step_info, trace

    def _get_multinomial_log_prob(self, feat_vec_dict, numerator_pos):
        """
        a single term in {eq:full_ordering}
        """
        theta_sums = [self.theta[feat_vec].sum() for feat_vec in feat_vec_dict.values()]
        multinomial_prob = self.theta[feat_vec_dict[numerator_pos]].sum() - scipy.misc.logsumexp(theta_sums)
        return multinomial_prob

    def _compute_log_probs(self, curr_order, gibbs_step_base=None, update_positions=None):
        """
        Driver to compute probabilities with some precompute if statements
        for speeding up computation.

        @params same as in _do_gibbs_sweep
        """
        if gibbs_step_base is None:
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
                update_steps=update_positions,
                base_feat_vec_dicts = gibbs_step_base.feat_vec_dicts,
                base_intermediate_seqs = gibbs_step_base.intermediate_seqs,
            )

        if gibbs_step_base is None:
            # Compute probabilities if they haven't been already
            log_probs = [
                self._get_multinomial_log_prob(feat_vec_dict_step, curr_mutate_pos)
                for curr_mutate_pos, feat_vec_dict_step in zip(curr_order, feat_vec_dicts)
            ]
        elif self.approx == "none":
            log_probs = list(gibbs_step_base.log_probs)
            for i in update_positions:
                log_probs[i] = self._get_multinomial_log_prob(feat_vec_dicts[i], curr_order[i])
        else:
            # Otherwise just reorder precomputed probabilities
            log_probs = [gibbs_step_base.log_probs[gibbs_step_base.order.index(item)] for item in curr_order]

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
