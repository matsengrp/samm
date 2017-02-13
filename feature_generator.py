import numpy as np
from common import get_theta_sum_mask

class FeatureGenerator:
    """
    Subclass this to have various types of feature vector generators.
    We assume all feature vectors are composed of ones and zeros.
    Each feature vector will a dictionary keyed on motifs with values being the
    corresponding index in the list of motifs (changing the right-hand bases
    first). We call these "sparse feature vectors".
    """

    def create_for_sequence(self, sequence, no_feat_vec_pos=[], do_feat_vec_pos=None):
        """
        @param sequence: current sequence (string)
        @param no_feat_vec_pos: do NOT generate feature vectors for these positions
        @param do_feat_vec_pos: do generate feature vectors for these positions.
                                By default, this will be set to all positions in the sequence.
        @return: list of sparse feature vectors

        If there are positions in both no_feat_vec_pos and do_feat_vec_pos, then
        no feature vector is generated for that position.
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

class ObservedSequenceMutationsFeatures:
    """
    An augmented ObservedSequenceMutations with features from before mutations have occured
    """
    def __init__(self, obs_seq_mutation, feat_matrix, feat_dict):
        self.feat_matrix = feat_matrix
        self.feat_dict = feat_dict

        # (shallow) copy over the rest of the data
        self.start_seq = obs_seq_mutation.start_seq
        self.end_seq = obs_seq_mutation.end_seq
        self.mutation_pos_dict = obs_seq_mutation.mutation_pos_dict
        self.num_mutations = obs_seq_mutation.num_mutations
        self.seq_len = obs_seq_mutation.seq_len

class FeatureMutationSteps:
    """
    Store all the features and other intermediate info at each mutation step
    """
    def __init__(self, seq_mut_order=None, intermediate_seqs=None, feature_vec_dicts=None, feat_matrix0=None, feat_matrix0T=None):
        """
        @param seq_mut_order: ImputedSequenceMutations[ObservedSequenceMutationsFeatures]
        """
        if seq_mut_order is not None:
            num_steps = seq_mut_order.obs_seq_mutation.num_mutations - 1
            self.intermediate_seqs = [seq_mut_order.obs_seq_mutation.start_seq] + [None] * num_steps
            self.feature_vec_dicts = [seq_mut_order.obs_seq_mutation.feat_dict] + [None] * num_steps
            self.feat_matrix0 = seq_mut_order.obs_seq_mutation.feat_matrix
            self.feat_matrix0T = self.feat_matrix0.transpose()
        elif intermediate_seqs is not None:
            self.intermediate_seqs = intermediate_seqs
            self.feature_vec_dicts = feature_vec_dicts
            self.feat_matrix0 = feat_matrix0
            self.feat_matrix0T = feat_matrix0T


    def copy(self):
        return FeatureMutationSteps(
            intermediate_seqs=list(self.intermediate_seqs),
            feature_vec_dicts=list(self.feature_vec_dicts),
            feat_matrix0=self.feat_matrix0,
            feat_matrix0T=self.feat_matrix0T,
        )

    def get_theta_sum(self, mutation_step, position, theta, col_idx=None):
        """
        @param position:
        """
        if col_idx is not None:
            return theta[self.feature_vec_dicts[mutation_step][position], col_idx].sum()
        else:
            mini_theta = theta[self.feature_vec_dicts[mutation_step][position], :]
            return mini_theta.sum(axis=0)

    def get_init_theta_sum(self, theta):
        return self.feat_matrix0.dot(theta)

    def update(self, i, seq_str, feat_dict):
        self.intermediate_seqs[i] = seq_str
        self.feature_vec_dicts[i] = feat_dict

    def __str__(self):
        return ",".join(self.intermediate_seqs)
