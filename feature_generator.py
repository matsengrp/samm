import numpy as np
from common import mutate_string

class FeatureGenerator:
    """
    Subclass this to have various types of feature vector generators
    We assume all feature vectors are composed of ones and zeros
    Each feature vector will just be the positions that are one.
    We call these "sparse feature vectors"
    """
    NUCLEOTIDES = "atcg"

    def generate(self, seq_mut_order):
        """
        @return: list of sparse feature vectors for positions in the risk group after the i-th mutation
        """
        raise NotImplementedError()

class SubmotifFeatureGenerator(FeatureGenerator):
    def __init__(self, submotif_len=3):
        assert(submotif_len % 2 == 1)
        self.submotif_len = 3
        self.flank_end_len = submotif_len/2
        self.feature_vec_len = np.power(4, submotif_len) + 1

    # TODO: make this a real feature generator
    def generate(self, seq_mut_order):
        """
        Feature vector are just indicators for which submotif. To get the index for a submotif
        indicator, vary the early positions fastest and later positions slowest.

        @param seq_mut_order: SequenceMutationOrder
        @return: list of sparse feature vectors for positions in the risk group after the i-th mutation
        """
        intermediate_seq = seq_mut_order.obs_seq_mutation.start_seq
        feature_vec_dicts = [
            self._create_feature_vec(intermediate_seq, [])
        ]
        for i, mutation_pos in enumerate(seq_mut_order.mutation_order):
            # update to get the new sequence after the i-th mutation
            intermediate_seq = mutate_string(
                intermediate_seq,
                mutation_pos,
                seq_mut_order.obs_seq_mutation.mutation_pos_dict[mutation_pos]
            )
            feature_vec_dicts.append(
                self._create_feature_vec(intermediate_seq, seq_mut_order.mutation_order[:i + 1])
            )
        return feature_vec_dicts

    def _create_feature_vec(self, intermediate_seq, no_feat_vec_pos):
        """
        @param no_feat_vec_pos: don't create feature vectors for these positions
        """
        feature_vec_dict = dict()
        for pos in range(len(intermediate_seq)):
            if pos in no_feat_vec_pos:
                # don't generate any feature vector for this position since it is not in the risk group
                continue
            if pos < self.flank_end_len or pos > self.flank_end_len + 1:
                # do special stuff cause positions are at the ends
                # TODO: update this. right now it sets all extreme positions to the same feature
                idx = self.feature_vec_len - 1
            else:
                submotif = intermediate_seq[pos - self.flank_end_len: pos + self.flank_end_len + 1]
                idx = 0
                for submotif_i, submotif_nuc in enumerate(submotif):
                    nuc_idx = FeatureGenerator.NUCLEOTIDES.index(submotif_nuc)
                    idx += nuc_idx * np.power(4, submotif_i)
            feature_vec_dict[pos] = [idx]
        return feature_vec_dict
