import numpy as np
from profile_support import profile
import scipy.sparse

class ObservedSequenceMutations:
    def __init__(self, start_seq, end_seq):
        assert(len(start_seq) == len(end_seq))

        self.start_seq = start_seq
        self.end_seq = end_seq

        # A dictionary with key as position and value as target nucleotide
        self.mutation_pos_dict = dict()
        for i in range(len(start_seq)):
            if start_seq[i] != end_seq[i]:
                self.mutation_pos_dict[i] = end_seq[i]
        self.num_mutations = len(self.mutation_pos_dict.keys())
        self.seq_len = len(start_seq)

    def __str__(self):
        return "Seq %s, Mutations %s" % (
            self.start_seq,
            self.mutation_pos_dict,
        )

class ImputedSequenceMutations:
    def __init__(self, obs_seq_mutation, mutation_order):
        """
        @param obs_seq_mutation: ObservedSequenceMutations
        @param mutation_order: a list of the positions in the order they mutated
        """
        self.obs_seq_mutation = obs_seq_mutation
        self.mutation_order = mutation_order

    def __str__(self):
        return "Seq %s, Mutation Order %s" % (
            self.obs_seq_mutation.start_seq,
            self.mutation_order,
        )

class FullSequenceMutations:
    def __init__(self, obs_seq_mutation, mutations):
        """
        @param obs_seq_mutation: ObservedSequenceMutations
        @param mutations: an ordered list of MutationPosTime
        """
        self.obs_seq_mutation = obs_seq_mutation
        self.mutations = mutations

    def get_mutation_order(self):
        return [m.pos for m in self.mutations]

    def __str__(self):
        return "%s => %s" % (
            self.obs_seq_mutation.start_seq,
            self.obs_seq_mutation.end_seq
        )

class MutationEvent:
    """
    Stores information on what happened during a mutation event
    """
    def __init__(self, time, pos, target_nucleotide):
        self.time = time
        self.pos = pos
        self.target_nucleotide = target_nucleotide

    def __str__(self):
        return "%d=%s (%.2g)" % (self.pos, self.target_nucleotide, self.time)

class SequencePositionFeatures:
    """
    ?
    """
    def __init__(self, seq_str, feat_matrix, is_sparse, active_rows, csr_matrix=None, feat_rows=None, feat_data=None):
        """
        @param active_rows: np boolean array indicating active rows
        @param theta_sums: np array with theta sums
        """
        self.seq_str = seq_str
        self.feat_matrix = feat_matrix
        self.csr_matrix = None
        if csr_matrix is not None:
            self.csr_matrix = csr_matrix
        self.active_rows = active_rows
        self.is_sparse = is_sparse
        self.theta_sums = None
        self.feat_rows = feat_rows
        self.feat_data = feat_data

    def get_active_positions(self):
        return np.nonzero(self.active_rows)[0]

    # @profile
    def update_feat_matrix(self, pos, feat_vec_idxs):
        if self.is_sparse:
            # We are manipulating the lil_matrix format
            self.feat_matrix.rows[pos] = feat_vec_idxs
            self.feat_matrix.data[pos] = [True] * len(feat_vec_idxs)

    @profile
    def update_theta_sums(self, theta, pos_idxs=None):
        if self.theta_sums is None:
            self.theta_sums = np.NaN * np.ones(self.active_rows.size)

        if self.is_sparse:
            if pos_idxs is None:
                # Update all the active positions
                if self.csr_matrix is None:
                    self.csr_matrix = scipy.sparse.csr_matrix(self.feat_matrix)
                self.theta_sums[self.active_rows] = self.csr_matrix[self.active_rows, :].dot(theta)
            else:
                # Update all the specified positions
                for p in pos_idxs:
                    self.theta_sums[p] = theta[self.feat_matrix.rows[p]].sum()
        else:
            raise NotImplementedError()

    @profile
    def copy(self):
        if self.is_sparse:
            if self.csr_matrix is not None:
                copy_feat_matrix = self.csr_matrix.tolil()
            else:
                copy_feat_matrix = scipy.sparse.lil_matrix(self.feat_matrix)
                # copy ONLY the wrapper around the matrix data, not the matrix data itself
                # a lot of the lists will stay the same, so we want to keep those untouched
                copy_feat_matrix.rows = self.feat_matrix.rows.copy()
                copy_feat_matrix.data = self.feat_matrix.data.copy()
        else:
            raise NotImplementedError()

        copy_feat = SequencePositionFeatures(
            self.seq_str,
            copy_feat_matrix,
            self.is_sparse,
            self.active_rows.copy(),
        )

        copy_feat.theta_sums = self.theta_sums.copy()
        return copy_feat

    def inactivate_position(self, position):
        self.active_rows[position] = 0

    def get_rows(self, pos_list):
        return np.array([pos_row_dict[p] for p in pos_list])

    def __str__(self):
        return "%s" % self.seq_str
