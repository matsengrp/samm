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
            seq_offset = (self.max_motif_len - feat_gen.motif_len)/2
            f_dict = feat_gen.create_for_sequence(seq_str, left_flank, right_flank, do_feat_vec_pos, offset=seq_offset)
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
        feat_mutation_steps = [MultiFeatureMutationStep() for i in range(seq_mut_order.obs_seq_mutation.num_mutations)]
        for offset, feat_gen in zip(self.feat_offsets, self.feat_gens):
            mut_steps = feat_gen.create_for_mutation_steps(seq_mut_order)
            for multi_f, single_f in zip(feat_mutation_steps, mut_steps):
                multi_f.update(single_f, offset)
        # for s in feat_mutation_steps:
        #     print "feat mut step", str(s)
        return feat_mutation_steps

    def get_shuffled_mutation_steps_delta(
        self,
        seq_mut_order,
        update_step,
        flanked_seq,
        already_mutated_pos,
    ):
        print "get_shuffled_mutation_steps_delta"
        first_mut_feats = []
        multi_feat_mut_step = MultiFeatureMutationStep()
        for offset, feat_gen in zip(self.feat_offsets, self.feat_gens):
            seq_offset = (seq_mut_order.obs_seq_mutation.motif_len - feat_gen.motif_len)/2
            mut_pos_feat, mut_step = feat_gen.get_shuffled_mutation_steps_delta(
                seq_mut_order,
                update_step,
                flanked_seq[seq_offset:-seq_offset] if seq_offset > 0 else flanked_seq,
                already_mutated_pos,
            )
            first_mut_feats.append(mut_pos_feat + offset)
            multi_feat_mut_step.update(mut_step, offset)
            print "mut_pos_feat", mut_pos_feat
            print "mut_step", mut_step
        print "first_mut_feats", first_mut_feats
        print "multi_feat_mut_step", multi_feat_mut_step
        return first_mut_feats, multi_feat_mut_step

    def create_remaining_mutation_steps(
        self,
        seq_mut_order,
        update_step_start,
    ):
        print "== create_remaining_mutation_steps =="
        feat_mutation_steps = [MultiFeatureMutationStep() for i in range(seq_mut_order.obs_seq_mutation.num_mutations - update_step_start)]
        for offset, feat_gen in zip(self.feat_offsets, self.feat_gens):
            mut_steps = feat_gen.create_remaining_mutation_steps(seq_mut_order, update_step_start)
            assert(len(mut_steps) == len(feat_mutation_steps))
            for multi_f, single_f in zip(feat_mutation_steps, mut_steps):
                print "single_f", single_f
                multi_f.update(single_f, offset)
        # for s in feat_mutation_steps:
        #     print "feat mut step", str(s)
        return feat_mutation_steps
