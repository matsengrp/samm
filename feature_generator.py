import numpy as np
from common import mutate_string

class FeatureGenerator:
    NUCLEOTIDES = "atcg"

    # We assume all feature vectors are composed of ones and zeros
    # Each feature vector will just be the positions that are one.
    # We call these "sparse feature vectors"

    @staticmethod
    # TODO: make this a real feature generator
    def generate(seq_mut_order, submotif_len=3):
        """
        Feature vector are just indicators for which submotif. To get the index for a submotif
        indicator, vary the early positions fastest and later positions slowest.

        @param seq_mut_order: SequenceMutationOrder
        @return: list of sparse feature vectors for positions in the risk group after the i-th mutation
        """
        assert(submotif_len % 2 == 1)
        def _create_feature_vec(intermediate_seq, no_feat_vec_pos):
            """
            @param no_feat_vec_pos: don't create feature vectors for these positions
            """
            flank_end_len = submotif_len/2
            feature_vec_len = np.power(4, submotif_len) + 1
            feature_vec_dict = dict()
            for pos in range(seq_mut_order.seq_len):
                if pos in no_feat_vec_pos:
                    # don't generate any feature vector for this position since it is not in the risk group
                    continue
                if pos < flank_end_len or pos > flank_end_len + 1:
                    # do special stuff cause positions are at the ends
                    # TODO: update this. right now it sets all extreme positions to the same feature
                    idx = feature_vec_len - 1
                else:
                    submotif = intermediate_seq[pos - flank_end_len: pos + flank_end_len + 1]
                    idx = 0
                    print "submotif", submotif
                    for submotif_i, submotif_nuc in enumerate(submotif):
                        print "submotif_nuc", submotif_nuc
                        nuc_idx = FeatureGenerator.NUCLEOTIDES.index(submotif_nuc)
                        print "submotif_i", submotif_i
                        print "nuc_idx", nuc_idx
                        idx += nuc_idx * np.power(4, submotif_i)
                feature_vec_dict[pos] = [idx]
            return feature_vec_dict

        intermediate_seq = seq_mut_order.obs_seq_mutation.start_seq
        feature_vec_dicts = [
            _create_feature_vec(intermediate_seq, [])
        ]
        for i, mutation_pos in enumerate(seq_mut_order.mutation_order):
            # update to get the new sequence after the i-th mutation
            intermediate_seq = mutate_string(intermediate_seq, mutation_pos, seq_mut_order.obs_seq_mutation.mutation_pos_dict[mutation_pos])
            feature_vec_dicts.append(
                _create_feature_vec(intermediate_seq, seq_mut_order.mutation_order[:i + 1])
            )
        return feature_vec_dicts
