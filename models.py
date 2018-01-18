from common import mutate_string, DEGENERATE_NUCLEOTIDE
import numpy as np

class ObservedSequenceMutations:
    def __init__(self, start_seq, end_seq, motif_len=3, left_flank_len=None, right_flank_len=None, collapse_list=[]):
        """
        @param start_seq: start sequence
        @param end_seq: ending sequence with mutations
        @param motif_len: needed to determine flanking ends/mutations to trim sequence
        @param left_flank_len: maximum left flank length for this motif length
        @param right_flank_len: maximum right flank length for this motif length
        @param collapse_list: list of tuples of (index offset, start index of run of "n"s, end index of run of "n"s) for bookkeeping later

        This class goes through half the sequence forward and finds the position where
        there are motif_len/2 conserved nucleotides, and does the same in reverse.

        Additionally, since we aren't generating features for any of the flanking nucleotides,
        it keeps an extra property self.flanks while saying that the start and end sequence
        are their corresponding subsequences after removing the flanks. This minimizes a lot
        of code rewriting and off-by-one errors with a minor increase in keeping around a little
        extra data.

        self.mutation_pos_dict is a dictionary with key as position and value as target nucleotide
        """

        assert(len(start_seq) == len(end_seq))
        self.motif_len = motif_len

        if left_flank_len is None:
            left_flank_len = motif_len/2
        if right_flank_len is None:
            right_flank_len = motif_len/2

        start_idx = 0
        end_idx = len(start_seq)

        skipped_left = 0
        skipped_right= 0

        # Go through half the sequence forward to find beginning conserved nucleotides
        # Also skip ns
        for flank_start_idx in range(len(start_seq)/2):
            if start_idx + left_flank_len == flank_start_idx:
                break
            elif start_seq[flank_start_idx] != end_seq[flank_start_idx] or \
                 start_seq[flank_start_idx] == DEGENERATE_NUCLEOTIDE or \
                 end_seq[flank_start_idx] == DEGENERATE_NUCLEOTIDE:
                start_idx = flank_start_idx + 1
                skipped_left += 1

        # Go through remaining half the sequence backward to find ending conserved nucleotides
        # Also skip ns
        for flank_end_idx in reversed(range(len(start_seq)/2, len(start_seq))):
            if end_idx - right_flank_len - 1 == flank_end_idx:
                break
            elif start_seq[flank_end_idx] != end_seq[flank_end_idx] or \
                 start_seq[flank_end_idx] == DEGENERATE_NUCLEOTIDE or \
                 end_seq[flank_end_idx] == DEGENERATE_NUCLEOTIDE:
                end_idx = flank_end_idx + 1
                skipped_right += 1

        self.left_flank = start_seq[start_idx:start_idx + left_flank_len]
        self.right_flank = start_seq[end_idx - right_flank_len:end_idx]

        start_seq = start_seq[start_idx + left_flank_len:end_idx - right_flank_len]
        end_seq = end_seq[start_idx + left_flank_len:end_idx - right_flank_len]

        self.mutation_pos_dict = dict()
        self.mutated_indicator = [0.] * len(start_seq)
        for i in range(len(start_seq)):
            if start_seq[i] != end_seq[i]:
                self.mutation_pos_dict[i] = end_seq[i]
                self.mutated_indicator[i] = 1.

        self.num_mutations = len(self.mutation_pos_dict.keys())
        self.left_flank_len = len(self.left_flank)
        self.skipped_mutations = skipped_left + skipped_right
        self.left_position_offset = left_flank_len + skipped_left
        self.right_position_offset = right_flank_len + skipped_right
        self.start_seq = start_seq
        self.start_seq_with_flanks = self.left_flank + start_seq + self.right_flank
        self.end_seq = end_seq
        self.end_seq_with_flanks = self.left_flank + end_seq + self.right_flank
        self.seq_len = len(self.start_seq)
        self.collapse_list = collapse_list
        assert(self.seq_len > 0)

    def set_start_feats(self, feat_matrix):
        self.feat_matrix_start = feat_matrix

    def __str__(self):
        return "Seq %s, Mutations %s" % (
            self.start_seq,
            self.mutation_pos_dict,
        )

class ImputedSequenceMutations:
    def __init__(self, obs_seq_mutation, mutation_order):
        """
        @param obs_seq_mutation: any object that needs to get augmented by a mutation order
                                (e.g. ObservedSequenceMutations or ObservedSequenceMutationsFeatures)
        @param mutation_order: a list of the positions in the order they mutated
        """
        self.obs_seq_mutation = obs_seq_mutation
        self.mutation_order = mutation_order

    def get_seq_at_step(self, step_idx, flanked=False):
        """
        @return the nucleotide sequence after the `step_idx`-th  mutation
        """
        intermediate_seq = self.obs_seq_mutation.start_seq
        for i in range(step_idx):
            mut_pos = self.mutation_order[i]
            intermediate_seq = mutate_string(
                intermediate_seq,
                mut_pos,
                self.obs_seq_mutation.end_seq[mut_pos]
            )
        if flanked:
            return self.obs_seq_mutation.left_flank + intermediate_seq + self.obs_seq_mutation.right_flank
        else:
            return intermediate_seq

    def __str__(self):
        return "Seq %s, Mutation Order %s" % (
            self.obs_seq_mutation.start_seq,
            self.mutation_order,
        )

class FullSequenceMutations:
    def __init__(self, start_seq, end_seq, left_flank, right_flank, mutations):
        """
        @param start_seq: processed start seq without flanks
        @param end_seq: processed end seq without flanks
        @param left_flank: left flank string
        @param right_flank: right flank string
        @param mutations: an ordered list of MutationPosTime
        """
        self.start_seq = start_seq
        self.end_seq = end_seq
        self.left_flank = left_flank
        self.right_flank = right_flank
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
