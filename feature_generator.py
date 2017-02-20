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
        @return: tuple of (FeatureMutationSteps, list of theta_sums)
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

class FeatureMutationStep:
    def __init__(self, mutating_pos_feat, neighbors_feat_old=dict(), neighbors_feat_new=dict()):
        self.mutating_pos_feat = mutating_pos_feat
        self.neighbors_feat_old = neighbors_feat_old
        self.neighbors_feat_new = neighbors_feat_new

    def __str__(self):
        return "(%d, old: %s, new: %s)" % (self.mutating_pos_feat, self.neighbors_feat_old, self.neighbors_feat_new)
