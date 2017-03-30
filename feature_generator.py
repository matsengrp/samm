import numpy as np

class FeatureGenerator:
    """
    Subclass this to have various types of feature vector generators.
    We assume all feature vectors are composed of ones and zeros.
    Each feature vector will a dictionary keyed on motifs with values being the
    corresponding index in the list of motifs (changing the right-hand bases
    first). We call these "sparse feature vectors".
    """

    def create_for_sequence(self, seq_str, do_feat_vec_pos=None):
        """
        @param sequence: current sequence (string)
        @param do_feat_vec_pos: do generate feature vectors for these positions.
                    By default, this will be set to all positions in the sequence.
        @return: list of sparse feature vectors
        """
        raise NotImplementedError()

    def create_for_mutation_steps(self, seq_mut_order):
        """
        @param seq_mut_order: a ImputedSequenceMutations
        @return: list of FeatureMutationStep
        """
        raise NotImplementedError()

    def update_for_mutation_steps(
        self,
        seq_mut_order,
        update_steps,
        base_feat_vec_dicts,
        base_intermediate_seqs,
    ):
        """
        Returns a feature vec given a list of steps that need to be updated
        Note: This makes a copy of the lists given so it won't modify `base_feat_vec_dicts, base_intermediate_seqs` in place.

        @param seq_mut_order: ImputedSequenceMutations
        @param update_steps: which steps in the mutation sequence are different and need to be updated
        @param base_feat_vec_dicts: list of feature vectors for positions from a similar mutation sequence
        @param base_intermediate_seqs: list of sequences from a similar mutation sequence
        @param base_feature_vec_theta_sums: list of theta sums from a similar mutation sequence
        @param theta: if a theta vector is given, function will also calculate the theta sum (theta * feature vector) for each position

        @return: 1. list of sparse feature vectors for positions in the risk group after the i-th mutation,
                2. list of intermediate sequences formed at each mutation step
                3. If theta and base_feature_vec_theta_sums given, list of theta sums for positions in the risk group at each mutation step. Otherwise None
        """
        raise NotImplementedError()

class MultiFeatureMutationStep:
    """
    Stores the deltas between each mutation step

    This allows fast calculation of the likelihood of the mutation order.
    Recall that the denominator = sum(exp(psi * theta)).
    At the next mutation step, we know that
    1. One of the terms must disappear because the risk group is smaller. One of the positions has mutated.
    2. The nucleotides next to the position that mutated have new motifs; so we need to update the motif
        indices for these positions. If we use the denominator from the previous step, we need to subtract
        out the old exp(psi * theta)) and add in new exp(psi * theta)
    """
    def __init__(self):
        """
        @param mutating_pos_feats: the feature index of the position that mutated
        @param neighbors_feat_old: the old feature indices of the positions next to the mutated position
        @param neighbors_feat_new: the new feature indices of the positions next to the mutated position
        """
        self.mutating_pos_feats = []
        self.neighbors_feat_old = dict()
        self.neighbors_feat_new = dict()

    def update(self, feat_mut_step, feature_offset):
        """
        @param feat_mut_step: FeatureMutationStep
        """
        self.mutating_pos_feats.append(feat_mut_step.mutating_pos_feat + feature_offset)
        self._merge_dicts(self.neighbors_feat_old, feat_mut_step.neighbors_feat_old, feature_offset)
        self._merge_dicts(self.neighbors_feat_new, feat_mut_step.neighbors_feat_new, feature_offset)

    def _merge_dicts(self, my_dict, new_dict, feature_offset):
        for k in new_dict.keys():
            new_feature = new_dict[k] + feature_offset
            if k not in my_dict:
                my_dict[k] = [new_feature]
            else:
                my_dict[k].append(new_feature)

    def __str__(self):
        return "(%s, old: %s, new: %s)" % (self.mutating_pos_feats, self.neighbors_feat_old, self.neighbors_feat_new)

class FeatureMutationStep:
    """
    Stores the deltas between each mutation step

    This allows fast calculation of the likelihood of the mutation order.
    Recall that the denominator = sum(exp(psi * theta)).
    At the next mutation step, we know that
    1. One of the terms must disappear because the risk group is smaller. One of the positions has mutated.
    2. The nucleotides next to the position that mutated have new motifs; so we need to update the motif
        indices for these positions. If we use the denominator from the previous step, we need to subtract
        out the old exp(psi * theta)) and add in new exp(psi * theta)
    """
    def __init__(self, mutating_pos_feat, neighbors_feat_old=dict(), neighbors_feat_new=dict()):
        """
        @param mutating_pos_feat: the feature index of the position that mutated
        @param neighbors_feat_old: the old feature indices of the positions next to the mutated position
        @param neighbors_feat_new: the new feature indices of the positions next to the mutated position
        """
        self.mutating_pos_feat = mutating_pos_feat
        self.neighbors_feat_old = neighbors_feat_old
        self.neighbors_feat_new = neighbors_feat_new

    def __str__(self):
        return "(%d, old: %s, new: %s)" % (self.mutating_pos_feat, self.neighbors_feat_old, self.neighbors_feat_new)
