from submotif_feature_generator import SubmotifFeatureGenerator

class HierarchicalMotifFeatureGenerator(FeatureGenerator):
    def __init__(self, motif_lens=[3,5]):
        self.feat_gens = [SubmotifFeatureGenerator(m_len) for m_len in motif_lens]
        feat_offsets = [feat_gen.feature_vec_len for feat_gen in self.feat_gens]
        self.feat_offsets = np.cumsum([0] + feat_offsets)[:-1]
        print "self.feat_offsets", self.feat_offsets

    def create_for_sequence(self, seq_str, left_flank, right_flank, do_feat_vec_pos=None):
        feat_vec_dict = None
        for offset, feat_gen in zip(self.feat_offsets, self.feat_gens):
            f_dict = feat_gen.create_for_sequence(seq_str, left_flank, right_flank, do_feat_vec_pos)
            if feat_vec_dict is None:
                feat_vec_dict = f_dict
            else:
                for pos, feat in f_dict.iteritems():
                    feat_vec_dict[pos].append(feat + self.feat_offsets)
        return feat_vec_dict
