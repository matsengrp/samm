import numpy as np
from common import mutate_string
from common import NUCLEOTIDES

class FeatureGenerator:
    """
    Subclass this to have various types of feature vector generators
    We assume all feature vectors are composed of ones and zeros
    Each feature vector will just be the positions that are one.
    We call these "sparse feature vectors"
    """

    def create_for_sequence(self, sequence, no_feat_vec_pos=[], do_feat_vec_pos=None):
        """
        @param sequence: current sequence
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
        @return: list of sparse feature vectors for all positions in the risk group after the i-th mutation,
                 for all mutation steps in seq_mut_order
        """
        raise NotImplementedError()

class SubmotifFeatureGenerator(FeatureGenerator):
    def __init__(self, submotif_len=3):
        assert(submotif_len % 2 == 1)
        self.submotif_len = 3
        self.flank_end_len = submotif_len/2
        self.feature_vec_len = np.power(4, submotif_len) + 1

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

    # TODO: make this a real feature generator
    def create_for_mutation_steps(self, seq_mut_order):
        """
        Feature vector are just indicators for which submotif. To get the index for a submotif
        indicator, vary the early positions fastest and later positions slowest.

        @param seq_mut_order: SequenceMutationOrder
        @return: list of sparse feature vectors for positions in the risk group after the i-th mutation
        """
        intermediate_seq = seq_mut_order.obs_seq_mutation.start_seq
        feature_vec_dicts = [
            self.create_for_sequence(intermediate_seq)
        ]
        for i, mutation_pos in enumerate(seq_mut_order.mutation_order):
            # update to get the new sequence after the i-th mutation
            intermediate_seq = mutate_string(
                intermediate_seq,
                mutation_pos,
                seq_mut_order.obs_seq_mutation.mutation_pos_dict[mutation_pos]
            )
            feature_vec_dicts.append(
                self.create_for_sequence(intermediate_seq, no_feat_vec_pos=seq_mut_order.mutation_order[:i + 1])
            )
        return feature_vec_dicts

    def _create_feature_vec_for_pos(self, pos, intermediate_seq):
        """
        @param no_feat_vec_pos: don't create feature vectors for these positions
        """
        if pos < self.flank_end_len or pos > self.flank_end_len + 1:
            # do special stuff cause positions are at the ends
            # TODO: update this. right now it sets all extreme positions to the same feature
            idx = self.feature_vec_len - 1
        else:
            submotif = intermediate_seq[pos - self.flank_end_len: pos + self.flank_end_len + 1]
            idx = 0
            for submotif_i, submotif_nuc in enumerate(submotif):
                nuc_idx = NUCLEOTIDES.index(submotif_nuc)
                idx += nuc_idx * np.power(4, submotif_i)
        return [idx]
