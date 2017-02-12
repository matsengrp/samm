import numpy as np
import scipy.misc
import logging as log

from models import ImputedSequenceMutations
from common import *

from sampler_collection import Sampler
from sampler_collection import SamplerResult
from profile_support import profile

from feature_generator import SubmotifFeatureGenerator

class GibbsStepInfo:
    def __init__(self, order, feat_vec_dicts, intermediate_seqs, feature_vec_theta_sums, log_probs):
        self.order = order
        self.feat_vec_dicts = feat_vec_dicts
        self.intermediate_seqs = intermediate_seqs
        self.feature_vec_theta_sums = feature_vec_theta_sums
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
        self.feature_vec_theta_sums_list = []

    def append(self, order, feat_vec_dicts, intermediate_seqs, feature_vec_theta_sums, log_probs, full_ordering_log_prob):
        """
        Add this new full ordering and its info to the list
        """
        self.order_list.append(order)
        self.feat_vec_dicts_list.append(feat_vec_dicts)
        self.intermediate_seqs_list.append(intermediate_seqs)
        self.feature_vec_theta_sums_list.append(feature_vec_theta_sums)
        self.log_probs_list.append(log_probs)
        self.full_ordering_log_probs.append(full_ordering_log_prob)

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
            self.feature_vec_theta_sums_list[sampled_idx],
            self.log_probs_list[sampled_idx],
        )

        sampled_order_probs = self.full_ordering_log_probs[sampled_idx]
        return gibbs_step_info, sampled_order_probs

class MutationOrderGibbsSampler(Sampler):
    """
    A class that will do the heavy lifting of Gibbs sampling.
    Returns an order and log probability vector (for tracing)
    """
    def run(self, init_order, burn_in, num_samples):
        assert(check_unordered_equal(init_order, self.mutated_positions))

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

    # @profile
    def _do_gibbs_sweep(self, curr_order, gibbs_step_info=None):
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
            pos_order_idx = curr_order.index(position)
            partial_order = curr_order[0:pos_order_idx] + curr_order[pos_order_idx + 1:]

            # Compute probabilities for the orderings under consideration
            # First consider the full ordering with position under consideration mutating last
            full_order_last = partial_order + [position]
            feat_vec_dicts, intermediate_seqs, log_probs, feature_vec_theta_sums = self._compute_log_probs(
                full_order_last,
                gibbs_step_info,
                update_positions=range(pos_order_idx, self.num_mutations),
            )

            gibbs_step_options = FullMutationOrderGibbsStepOptions()
            gibbs_step_options.append(
                full_order_last,
                feat_vec_dicts,
                intermediate_seqs,
                feature_vec_theta_sums,
                log_probs,
                log_probs.sum(),
            )

            # iterate through the rest of the possible full mutation orders consistent with this partial order
            prev_full_order = full_order_last
            prev_feature_vec_theta_sums = feature_vec_theta_sums
            for idx, i in enumerate(reversed(range(self.num_mutations - 1))):
                prev_feat_dicts = feat_vec_dicts
                if i == self.num_mutations - 2:
                    # If the last mutation, only need to update one feature vec
                    update_steps = [i]
                else:
                    # If not the last mutation, update two feature vecs
                    update_steps = [i, i+1]
                possible_full_order = partial_order[:i] + [position] + partial_order[i:]
                feat_vec_dicts, intermediate_seqs, feature_vec_theta_sums = self.feature_generator.update_for_mutation_steps(
                    ImputedSequenceMutations(
                        self.obs_seq_mutation,
                        possible_full_order
                    ),
                    update_steps=update_steps,
                    base_feat_vec_dicts = feat_vec_dicts,
                    base_intermediate_seqs = intermediate_seqs,
                    base_feature_vec_theta_sums = feature_vec_theta_sums,
                    theta=self.theta,
                )

                # multiply the sequence of multinomials to get the probability of the full ordering
                # the product in {eq:full_ordering}
                if self.approx == 'none':
                    log_probs = self._update_log_prob_from_shuffle(i, log_probs, possible_full_order, prev_full_order, feature_vec_theta_sums, prev_feature_vec_theta_sums)
                    full_ordering_log_prob = log_probs.sum()
                else:
                    # Stupid speed up #2:
                    # The change in theta doesn't really change the i+1 element
                    # of our multinomial vector from the previous ith element:
                    full_ordering_log_prob = gibbs_step_options.full_ordering_log_probs[-1] + \
                        self.theta[feature_vec_theta_sums[i][possible_full_order[i]]].sum() - \
                        self.theta[feat_vec_dicts[i][full_order_last[i]]].sum() + \
                        log_probs[i] - log_probs[i + 1]

                prev_full_order = possible_full_order
                prev_feature_vec_theta_sums = feature_vec_theta_sums

                gibbs_step_options.append(
                    possible_full_order,
                    feat_vec_dicts,
                    intermediate_seqs,
                    feature_vec_theta_sums,
                    log_probs,
                    full_ordering_log_prob,
                )

            # now perform a draw from the multinomial distribution of full orderings
            # the multinomial follows the distribution in {eq:order_conditional_prob}
            gibbs_step_info, curr_order_probs = gibbs_step_options.sample()
            curr_order = gibbs_step_info.order

            # Output all log probabilities for trace
            trace.append(curr_order_probs)

        return gibbs_step_info, trace

    # @profile
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
        log_probs[i] += feature_vec_theta_sums[i][pos_mutate_earlier] - feature_vec_theta_sums[i][pos_mutate_later]

        # Term at `i + 1` is complicated: we need to change out the numerator and some terms in the denominator
        # Remember that the denominator is the sum of exp thetas - some of the thetas are have the wrong value if the mutation order is shuffled
        # We only chose the theta value that are potentially wrong.
        # We are making the assumption that the features only change for positions close to the positions that got shuffled.
        changed_feature_positions = set(
            range(max(pos_mutate_earlier - self.motif_len/2, 0), min(pos_mutate_earlier + self.motif_len/2 + 1, self.seq_len))
            + range(max(pos_mutate_later - self.motif_len/2, 0), min(pos_mutate_later + self.motif_len/2 + 1, self.seq_len))
        ) - set(order[:i])

        old_sumexp = np.exp(prev_feature_vec_theta_sums[i + 1][pos_mutate_earlier] - log_probs[i + 1])

        old_sumexp_terms = sum([
            np.exp(prev_feature_vec_theta_sums[i + 1][p]) for p in changed_feature_positions if p != pos_mutate_later
        ])
        new_sumexp_terms = sum(np.exp([feature_vec_theta_sums[i + 1][p] for p in changed_feature_positions if p != pos_mutate_earlier]))

        log_probs[i + 1] = feature_vec_theta_sums[i + 1][pos_mutate_later] - np.log(
            old_sumexp + new_sumexp_terms - old_sumexp_terms
        )

        return log_probs


    # @profile
    def _get_multinomial_log_prob(self, numerator_pos, feature_vec_theta_sum):
        """
        a single term in {eq:full_ordering}
        """
        theta_sums = feature_vec_theta_sum.values()
        multinomial_prob = feature_vec_theta_sum[numerator_pos] - scipy.misc.logsumexp(theta_sums)
        return multinomial_prob

    @profile
    def _compute_log_probs(self, curr_order, gibbs_step_base=None, update_positions=None):
        """
        Driver to compute probabilities with some precompute if statements
        for speeding up computation.

        @params same as in _do_gibbs_sweep
        """
        if gibbs_step_base is None:
            # Compute dictionaries if we haven't yet
            feat_vec_things = self.feature_generator.create_for_mutation_steps(
                ImputedSequenceMutations(
                    self.obs_seq_mutation,
                    curr_order,
                ),
                self.theta
            )
            bad_feature_generator = SubmotifFeatureGenerator(motif_len = 3)
            feat_vec_dicts, intermediate_seqs, feature_vec_theta_sums = bad_feature_generator.create_for_mutation_steps(
                ImputedSequenceMutations(
                    self.obs_seq_mutation,
                    curr_order,
                ),
                self.theta
            )
            1/0
        else:
            # Update dictionaries if they've been computed already
            feat_vec_dicts, intermediate_seqs, feature_vec_theta_sums = self.feature_generator.update_for_mutation_steps(
                ImputedSequenceMutations(
                    self.obs_seq_mutation,
                    curr_order,
                ),
                update_steps=update_positions,
                base_feat_vec_dicts = gibbs_step_base.feat_vec_dicts,
                base_intermediate_seqs = gibbs_step_base.intermediate_seqs,
                base_feature_vec_theta_sums = gibbs_step_base.feature_vec_theta_sums,
                theta=self.theta,
            )

        if gibbs_step_base is None:
            # Compute probabilities if they haven't been already
            log_probs = np.array([
                self._get_multinomial_log_prob(curr_mutate_pos, theta_sum)
                for curr_mutate_pos, theta_sum in zip(curr_order, feature_vec_theta_sums)
            ])
        elif self.approx == "none":
            log_probs = np.copy(gibbs_step_base.log_probs)
            for i in update_positions:
                log_probs[i] = self._get_multinomial_log_prob( curr_order[i], feature_vec_theta_sums[i])
        else:
            # Otherwise just reorder precomputed probabilities
            log_probs = np.array([gibbs_step_base.log_probs[gibbs_step_base.order.index(item)] for item in curr_order])

        return feat_vec_dicts, intermediate_seqs, log_probs, feature_vec_theta_sums

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
