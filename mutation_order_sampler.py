import numpy as np

class MutationOrderSampler:
    def __init__(self, theta, feature_generator):
        self.theta = theta
        self.feature_generator = feature_generator

    def sample_order(self, obs_seq_mutation):
        """
        @param obs_seq_mutation: a ObservedSequenceMutations to know the starting and ending sequence
        @return a sampled mutation order with model theta, return type ImputedSequenceMutations
        """
        unmutated_positions = obs_seq_mutation.mutation_pos_dict.keys()
        intermediate_seq = obs_seq_mutation.start_seq
        num_pos_to_mutate = len(unmutated_positions)
        mutation_order = []
        for step_i in range(num_pos_to_mutate):
            feature_vecs = self.feature_generator.create_for_sequence(
                intermediate_seq,
                do_feat_vec_pos=unmutated_positions,
            )
            multinomial_weights = np.array([np.exp(np.sum(self.theta[feature_vecs[pos]])) for pos in unmutated_positions])
            multinomial_weights = multinomial_weights/np.sum(multinomial_weights)
            multinomial_sample = np.random.multinomial(1, multinomial_weights)
            sampled_idx = np.where(multinomial_sample == 1)[0][0]
            mutated_pos = unmutated_positions.pop(sampled_idx)
            mutation_order.append(mutated_pos)
        return mutation_order
