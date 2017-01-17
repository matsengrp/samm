import numpy as np
from models import ImputedSequenceMutations
from common import sample_multinomial

class MutationOrderGibbsSampler:
    def __init__(self, theta, feature_generator, obs_seq_mutation):
        """
        @param theta: numpy vector of model parameters
        @param feature_generator: FeatureGenerator
        @param obs_seq_mutation: a ObservedSequenceMutations to know the starting and ending sequence
        """
        self.theta = theta
        self.feature_generator = feature_generator
        self.obs_seq_mutation = obs_seq_mutation
        self.mutated_positions = obs_seq_mutation.mutation_pos_dict.keys()
        self.num_mutations = len(self.mutated_positions)
        self.samples = []

    def sample_burn_in(self, init_order, burn_in_sweeps):
        curr_order = init_order
        for i in range(burn_in_sweeps):
            sample = self._do_gibbs_sweep(curr_order)
        self.samples = [sample]

    def sample_orders(self, num_samples):
        """
        @param num_samples: number of samples needed
        @return a list of sampled mutation order with model theta
        The gibbs sampler will start from its latest sample
        """
        if len(self.samples) == 0:
            raise ValueError("Please do burn-in first")

        curr_order = self.samples[-1]
        for i in range(num_samples):
            curr_order = self._do_gibbs_sweep(curr_order)
            self.samples.append(curr_order)
        return self.samples[-num_samples:]

    def _do_gibbs_sweep(self, curr_order):
        """
        One gibbs sweep is a gibbs sampling step for all the positions
        """
        # sample full ordering from conditional prob for this position
        for position in self.mutated_positions:
            pos_order_idx = curr_order.index(position)
            partial_order = curr_order[0:pos_order_idx] + curr_order[pos_order_idx + 1:]

            # the probabilities of each full ordering
            full_ordering_probs = []
            # the orderings under consideration
            # TODO: we can get rid of this variable and just recompute the order
            # It's here right now cause it makes life easy
            full_orderings = []

            # first consider the full ordering with the mutate in the last position
            full_order_last = partial_order + [position]
            feat_vec_dicts = self.feature_generator.create_for_mutation_steps(
                ImputedSequenceMutations(
                    self.obs_seq_mutation,
                    full_order_last,
                )
            )
            multinomial_sequence = [
                self._get_multinomial_prob(feat_vec_dict_step, curr_mutate_pos)
                for curr_mutate_pos, feat_vec_dict_step in zip(full_order_last, feat_vec_dicts)
            ]
            full_orderings.append(full_order_last)
            full_ordering_probs.append(np.prod(multinomial_sequence))

            # iterate through the rest of the possible full mutation orders consistent with this partial order
            for i in reversed(range(self.num_mutations - 1)):
                possible_full_order = partial_order[:i] + [position] + partial_order[i:]
                feat_vec_dicts = self.feature_generator.create_for_mutation_steps(
                    ImputedSequenceMutations(
                        self.obs_seq_mutation,
                        possible_full_order
                    )
                )
                # calculate multinomial probs - only need to update 2 values (the one where this position mutates and
                # the position that mutates right after it), rest are the same
                multinomial_sequence[i] = self._get_multinomial_prob(feat_vec_dicts[i], possible_full_order[i])
                multinomial_sequence[i + 1] = self._get_multinomial_prob(feat_vec_dicts[i + 1], possible_full_order[i + 1])

                full_orderings.append(possible_full_order)
                # multiply the sequence of multinomials to get the probability of the full ordering
                full_ordering_probs.append(np.prod(multinomial_sequence))

            # now perform a draw from the multinomial distribution of full orderings
            sampled_order_idx = sample_multinomial(full_ordering_probs)
            # update the ordering
            curr_order = full_orderings[sampled_order_idx]
        return curr_order

    def _get_multinomial_prob(self, feat_vec_dict, numerator_pos):
        numerator = np.exp(np.sum(self.theta[feat_vec_dict[numerator_pos]]))
        denominator = np.sum([
            np.exp(np.sum(self.theta[feat_vec])) for feat_vec in feat_vec_dict.values()
        ])
        return numerator / denominator
