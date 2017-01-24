import numpy as np
from common import mutate_string
from common import NUCLEOTIDES

class FeatureGenerator:
    """
    Subclass this to have various types of feature vector generators.
    We assume all feature vectors are composed of ones and zeros.
    Each feature vector will a dictionary keyed on motifs with values being the
    corresponding index in the list of motifs (changing the left-hand bases
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

    # TODO: make this a real feature generator (deal with ends properly)
    def create_for_mutation_steps(self, seq_mut_order):
        """
        Feature vector are just indicators for which submotif. To get the index for a submotif
        indicator, vary the early positions fastest and later positions slowest.

        @param seq_mut_order: ImputedSequenceMutations
        @return: list of sparse feature vectors for positions in the risk group after the i-th mutation
        """
        intermediate_seq = seq_mut_order.obs_seq_mutation.start_seq
        feature_vec_dicts = [
            self.create_for_sequence(intermediate_seq)
        ]
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
                    feat_vec_dict_update[p] = feature_vec_dicts[-1][p]

            feature_vec_dicts.append(feat_vec_dict_update)

        return feature_vec_dicts

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
            idx = 0
            for submotif_i, submotif_nuc in enumerate(submotif):
                nuc_idx = NUCLEOTIDES.index(submotif_nuc)
                idx += nuc_idx * np.power(4, submotif_i)
        return [idx]

    @staticmethod
    def get_motif_list():
        motif_list = []
        for i in NUCLEOTIDES:
            for j in NUCLEOTIDES:
                for k in NUCLEOTIDES:
                    motif = "%s%s%s" % (k, j, i)
                    motif_list.append(motif)
        return motif_list
