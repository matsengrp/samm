from feature_generator import *
from submotif_feature_generator import SubmotifFeatureGenerator
from scipy.sparse import hstack

class HierarchicalMotifFeatureGenerator(FeatureGenerator):
    def __init__(self, motif_lens=[3,5]):
        self.motif_lens = motif_lens
        self.max_motif_len = max(motif_lens)
        self.motif_len = self.max_motif_len

        self.feat_gens = [SubmotifFeatureGenerator(m_len) for m_len in motif_lens]
        feat_offsets = [feat_gen.feature_vec_len for feat_gen in self.feat_gens]
        self.feat_offsets = np.cumsum([0] + feat_offsets)[:-1]
        self.feature_vec_len = np.sum(feat_offsets)
        self.motif_list = []
        for f in self.feat_gens:
            self.motif_list += f.motif_list

    def create_for_sequence(self, seq_str, left_flank, right_flank, do_feat_vec_pos=None):
        feat_vec_dict = dict()
        for offset, feat_gen in zip(self.feat_offsets, self.feat_gens):
            offset = self.max_motif_len - feat_gen.motif_len
            mini_seq_str = seq_str[offset/2:-offset/2]
            mini_left = left_flank[offset/2:] if offset > 1
            mini_right = left_flank[offset/2:] if offset < 1
            f_dict = feat_gen.create_for_sequence(mini_seq_str, left_flank[offset/2:], right_flank[:-offset/2], do_feat_vec_pos)
            for pos, feat in f_dict.iteritems():
                if pos in feat_vec_dict:
                    feat_vec_dict[pos].append(feat + offset)
                else:
                    feat_vec_dict[pos] = [feat + offset]
        return feat_vec_dict

    def create_base_features(self, obs_seq_mutation):
        feat_vec_dict = dict()
        feat_mats = []
        for offset, feat_gen in zip(self.feat_offsets, self.feat_gens):
            # HACK
            offset = self.max_motif_len - feat_gen.motif_len
            if offset
            obs_seq_mutation = ObservedSequenceMutations(obs_seq_mutation.start_seq[offset:-offset])
            f_dict, feat_mat = feat_gen._get_base_features(obs_seq_mutation)
            for pos, feat in f_dict.iteritems():
                if pos in feat_vec_dict:
                    feat_vec_dict[pos].append(feat + offset)
                else:
                    feat_vec_dict[pos] = [feat + offset]
            feat_mats.append(feat_mat)
        full_feat_mat = hstack(feat_mats, format="csr")
        obs_seq_mutation.set_start_feats(feat_vec_dict, full_feat_mat)
        return obs_seq_mutation

    def create_for_mutation_steps(self, seq_mut_order):
        for offset, feat_gen in zip(self.feat_offsets, self.feat_gens):
