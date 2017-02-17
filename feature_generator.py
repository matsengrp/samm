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

class ObservedSequenceMutationsFeatures:
    """
    An augmented ObservedSequenceMutations with features from before mutations have occured
    """
    def __init__(self, obs_seq_mutation, feat_matrix, feat_dict):
        """
        @param feat_matrix: a scipy sparse csr_matrix, the features before any mutations have occurred
        @param feat_dict: a dictionary of our own implementation of feature vectors, the features before any mutations have occurred
        """
        self.feat_matrix = feat_matrix
        self.feat_dict = feat_dict

        # (shallow) copy over the rest of the data
        self.start_seq = obs_seq_mutation.start_seq
        self.end_seq = obs_seq_mutation.end_seq
        self.left_flank = obs_seq_mutation.left_flank
        self.right_flank = obs_seq_mutation.right_flank
        self.mutation_pos_dict = obs_seq_mutation.mutation_pos_dict
        self.num_mutations = obs_seq_mutation.num_mutations
        self.seq_len = obs_seq_mutation.seq_len

class FeatureMutationSteps:
    """
    Store all the features and other intermediate info at each mutation step
    """
    def __init__(self, seq_mut_order=None, intermediate_seqs=None, feature_vec_dicts=None, feat_matrix0=None, feat_matrix0T=None):
        """
        Two options for instantiating this object:
        Option 1:
        @param seq_mut_order: ImputedSequenceMutations[ObservedSequenceMutationsFeatures]

        Option 2:
        @param intermediate_seqs: list of strings at each mutation step
        @param feature_vec_dicts: list of dictionaries with our own sparse feature vector representation
        @param feat_matrix0: scipy sparse csr_matrix representation of the features at each position before any mutations have occurred
        @param feat_matrix0T: the transpose of feat_matrix0
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
        """
        Creates a semi-shallow copy of this object
        """
        return FeatureMutationSteps(
            intermediate_seqs=list(self.intermediate_seqs),
            feature_vec_dicts=list(self.feature_vec_dicts),
            feat_matrix0=self.feat_matrix0,
            feat_matrix0T=self.feat_matrix0T,
        )

    def get_theta_sum(self, mutation_step, position, theta, col_idx=None):
        """
        Calculate feature vector * theta for a given mutation step
        @param mutation_step: index of the mutation step
        @param position: the position to calculate the theta sum for
        @param theta: numpy vector
        @param col_idx: if provided, only calculates the theta sum for that column
        """
        if col_idx is not None:
            return theta[self.feature_vec_dicts[mutation_step][position], col_idx].sum()
        else:
            mini_theta = theta[self.feature_vec_dicts[mutation_step][position], :]
            return mini_theta.sum(axis=0)

    def get_init_theta_sum(self, theta):
        """
        Multiply theta with our initial matrix
        """
        return self.feat_matrix0.dot(theta)

    def update(self, mutation_step, seq_str, feat_dict):
        """
        Update the state of a mutation step
        @param mutation_step: the index of the mutation step to be updated
        @param seq_str: string
        @param feat_dict: a dictionary with our own sparse feature implementation
        """
        self.intermediate_seqs[mutation_step] = seq_str
        self.feature_vec_dicts[mutation_step] = feat_dict

    def __str__(self):
        return ",".join(self.intermediate_seqs)
