import time
import numpy as np
from common import mutate_string
from common import NUCLEOTIDES
import itertools

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
        @return: a tuple with:
            1. list of sparse feature vectors for all positions in the risk group after the i-th mutation,
                for all mutation steps in seq_mut_order
            2. a list of all the intermediate sequences that occured throughout the mutation steps
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
        Calculates a list of feature vectors and intermediate sequences by borrowoing information from
        previously calculated feature vectors and intermediate sequences. By indicating after which mutation
        steps the intermediate sequences differ, we can know which steps need to be calculated vs. which steps
        can just be copied.

        @param seq_mut_order: a ImputedSequenceMutations
        @param update_steps: the mutation steps such that the intermediate sequence was different
                            after these steps were different from the mutation steps used to create
                                `base_feat_vec_dicts` and `base_intermediate_seqs`
        @param base_feat_vec_dicts: a list of sparse feature vectors for all mutation steps
        @param base_intermediate_seqs: a list of all the intermediate sequences for all mutation steps
        @return: a tuple with:
            1. list of sparse feature vectors for all positions in the risk group after the i-th mutation,
                for all mutation steps in seq_mut_order
            2. a list of all the intermediate sequences that occured throughout the mutation steps
        """
        raise NotImplementedError()

class SubmotifFeatureGenerator(FeatureGenerator):
    """
    This makes motifs of the same length (must be odd). For positions on the edge, we currently
    just use an indicator to denote the position is on the edge.
    TODO: Do something else for edge positions?
    """
    def __init__(self, submotif_len=3):
        assert(submotif_len % 2 == 1)
        self.submotif_len = submotif_len
        self.flank_end_len = submotif_len/2
        self.feature_vec_len = np.power(4, submotif_len) + 1

        motif_list = self.get_motif_list(submotif_len)
        self.motif_dict = {motif: i for i, motif in enumerate(motif_list)}

    def create_for_sequence(self, sequence, no_feat_vec_pos=[], do_feat_vec_pos=None):
        feature_vec_dict = dict()
        if do_feat_vec_pos is None:
            do_feat_vec_pos = range(len(sequence))

        for pos in do_feat_vec_pos:
            if pos in no_feat_vec_pos:
                # don't generate any feature vector for this position since it is not in the risk group
                continue
            feature_vec_dict[pos] = self._create_feature_vec_for_pos(pos, sequence)

        return feature_vec_dict

    # TODO: make this a real feature generator (deal with ends properly)
    def create_for_mutation_steps(self, seq_mut_order):
        """
        @param seq_mut_order: ImputedSequenceMutations
        @return: list of sparse feature vectors for positions in the risk group after the i-th mutation
        """
        num_steps = seq_mut_order.obs_seq_mutation.num_mutations - 1
        intermediate_seq = seq_mut_order.obs_seq_mutation.start_seq
        intermediate_seqs = [intermediate_seq] + [None] * num_steps
        feature_vec_dicts = [self.create_for_sequence(intermediate_seq)] + [None] * num_steps

        for i, mutation_pos in enumerate(seq_mut_order.mutation_order[:-1]):
            # update to get the new sequence after the i-th mutation
            intermediate_seq = mutate_string(
                intermediate_seq,
                mutation_pos,
                seq_mut_order.obs_seq_mutation.mutation_pos_dict[mutation_pos]
            )
            # Get the feature vectors for the positions that might be affected by the latest mutation
            # Don't calculate feature vectors for positions that have mutated already
            no_feat_vec_pos = seq_mut_order.mutation_order[:i + 1]
            # Calculate feature vectors for positions that are close to the mutation
            do_feat_vec_pos = range(
                max(mutation_pos - self.flank_end_len, 0),
                min(mutation_pos + self.flank_end_len, seq_mut_order.obs_seq_mutation.seq_len),
            )
            feat_vec_dict_update = self.create_for_sequence(
                intermediate_seq,
                no_feat_vec_pos=no_feat_vec_pos,
                do_feat_vec_pos=do_feat_vec_pos,
            )
            # Populate rest of dict with the previously calculated feature vectors
            # TODO: maybe we can copy the dict faster
            for p in range(seq_mut_order.obs_seq_mutation.seq_len):
                if p not in no_feat_vec_pos and p not in do_feat_vec_pos:
                    feat_vec_dict_update[p] = feature_vec_dicts[i][p]

            feature_vec_dicts[i + 1] = feat_vec_dict_update
            intermediate_seqs[i + 1] = intermediate_seq
        return feature_vec_dicts, intermediate_seqs

    # TODO: make this a real feature generator (deal with ends properly)
    def update_for_mutation_steps(self,
        seq_mut_order,
        update_steps,
        base_feat_vec_dicts,
        base_intermediate_seqs,
    ):
        """
        Returns a feature vec Given a list of steps that need to be updated

        @param seq_mut_order: ImputedSequenceMutations
        @return: list of sparse feature vectors for positions in the risk group after the i-th mutation
        """
        intermediate_seqs = list(base_intermediate_seqs)
        feature_vec_dicts = list(base_feat_vec_dicts)
        for i in update_steps:
            mutation_pos = seq_mut_order.mutation_order[i]
            # update to get the new sequence after the i-th mutation
            intermediate_seqs[i+1] = mutate_string(
                intermediate_seqs[i],
                mutation_pos,
                seq_mut_order.obs_seq_mutation.mutation_pos_dict[mutation_pos]
            )
            # Get the feature vectors for the positions that might be affected by the latest mutation
            # Don't calculate feature vectors for positions that have mutated already
            no_feat_vec_pos = seq_mut_order.mutation_order[:i + 1]
            # Calculate feature vectors for positions that are close to the mutation
            do_feat_vec_pos = range(
                max(mutation_pos - self.flank_end_len, 0),
                min(mutation_pos + self.flank_end_len, seq_mut_order.obs_seq_mutation.seq_len),
            )
            feat_vec_dict_update = self.create_for_sequence(
                intermediate_seqs[i + 1],
                no_feat_vec_pos=no_feat_vec_pos,
                do_feat_vec_pos=do_feat_vec_pos,
            )
            # Populate rest of dict with the previously calculated feature vectors
            # TODO: maybe we can copy the dict faster
            for p in range(seq_mut_order.obs_seq_mutation.seq_len):
                if p not in no_feat_vec_pos and p not in do_feat_vec_pos:
                    feat_vec_dict_update[p] = feature_vec_dicts[i][p]

            feature_vec_dicts[i + 1] = feat_vec_dict_update
        return feature_vec_dicts, intermediate_seqs

    def _create_feature_vec_for_pos(self, pos, intermediate_seq):
        """
        @param no_feat_vec_pos: don't create feature vectors for these positions
        """
        if pos < self.flank_end_len or pos > len(intermediate_seq) - 1 - self.flank_end_len:
            # do special stuff cause positions are at the ends
            # TODO: update this. right now it sets all extreme positions to the same feature
            idx = self.feature_vec_len - 1
        else:
            submotif = intermediate_seq[pos - self.flank_end_len: pos + self.flank_end_len + 1]
            idx = self.motif_dict[submotif]
        return [idx]

    @staticmethod
    def get_motif_list(motif_len):
        motif_list = itertools.product(*([NUCLEOTIDES] * motif_len))
        return ["".join(m) for m in motif_list]
