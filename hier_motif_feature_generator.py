from feature_generator import *
from submotif_feature_generator import SubmotifFeatureGenerator
from scipy.sparse import hstack
from itertools import product

class HierarchicalMotifFeatureGenerator(FeatureGenerator):
    def __init__(self, motif_lens=[3,5], left_motif_flank_len_list=None):
        """
        @param motif_lens: list of odd-numbered motif lengths
        @param left_motif_flank_len_list: list of lengths of left motif flank; 0 will mutate the leftmost position, 1 the next to left, etc.
        """

        self.motif_lens = motif_lens

        self.max_motif_len = max(motif_lens)
        self.motif_len = self.max_motif_len

        if left_motif_flank_len_list is None:
            # default to central base mutating
            left_motif_flank_len_list = []
            for motif_len in motif_lens:
                left_motif_flank_len_list.append([motif_len/2])

        # Find the maximum left and right motif flank lengths to pass to SubmotifFeatureGenerator
        # in order to update all the relevant features
        for motif_len, left_flanks in zip(self.motif_lens, left_motif_flank_len_list):
            if motif_len == self.max_motif_len:
                self.max_left_motif_flank_len = max(left_flanks)
                self.max_right_motif_flank_len = self.motif_len - min(left_flanks) - 1

        # Create list of feature generators for different motif lengths and different flank lengths
        self.feat_gens = []
        for motif_len, left_motif_flank_lens in zip(motif_lens, left_motif_flank_len_list):
            for left_motif_flank_len in left_motif_flank_lens:
                self.feat_gens.append(
                        SubmotifFeatureGenerator(
                            motif_len=motif_len,
                            left_motif_flank_len=left_motif_flank_len,
                            hier_offset=self.max_left_motif_flank_len - left_motif_flank_len,
                            left_update_region=self.max_left_motif_flank_len,
                            right_update_region=self.max_right_motif_flank_len,
                        )
                    )

        feat_offsets = [feat_gen.feature_vec_len for feat_gen in self.feat_gens]
        self.feat_offsets = np.cumsum([0] + feat_offsets)[:-1]

        self.feature_vec_len = np.sum(feat_offsets)
        self.motif_list = []
        self.mutating_pos_list = []
        for f in self.feat_gens:
            self.motif_list += f.motif_list
            # pass mutating positions along with motifs
            self.mutating_pos_list += [f.left_motif_flank_len] * len(f.motif_list)

    def create_for_sequence(self, seq_str, left_flank, right_flank, do_feat_vec_pos=None):
        feat_vec_dict = dict()
        for offset, feat_gen in zip(self.feat_offsets, self.feat_gens):
            f_dict = feat_gen.create_for_sequence(seq_str, left_flank, right_flank, do_feat_vec_pos)
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

    def count_mutated_motifs(self, seq_mut_order):
        mutated_motifs = []
        for offset, feat_gen in zip(self.feat_offsets, self.feat_gens):
            curr_mutated_motifs = feat_gen.count_mutated_motifs(seq_mut_order)
            mutated_motifs.append(curr_mutated_motifs)
        return mutated_motifs

    def create_for_mutation_steps(self, seq_mut_order):
        feat_mutation_steps = [MultiFeatureMutationStep() for i in range(seq_mut_order.obs_seq_mutation.num_mutations)]
        for offset, feat_gen in zip(self.feat_offsets, self.feat_gens):
            mut_steps = feat_gen.create_for_mutation_steps(seq_mut_order)
            for multi_f, single_f in zip(feat_mutation_steps, mut_steps):
                multi_f.update(single_f, offset)
        return feat_mutation_steps

    def get_shuffled_mutation_steps_delta(
        self,
        seq_mut_order,
        update_step,
        flanked_seq,
        already_mutated_pos,
    ):
        first_mut_feats = []
        multi_feat_mut_step = MultiFeatureMutationStep()
        for offset, feat_gen in zip(self.feat_offsets, self.feat_gens):
            # determine where the left flank should start for this particular feature generator
            left_flank_start = len(seq_mut_order.obs_seq_mutation.left_flank) - feat_gen.left_motif_flank_len
            mut_pos_feat, mut_step = feat_gen.get_shuffled_mutation_steps_delta(
                seq_mut_order,
                update_step,
                flanked_seq[left_flank_start:]
                already_mutated_pos,
            )
            first_mut_feats.append(mut_pos_feat + offset)
            multi_feat_mut_step.update(mut_step, offset)
        return first_mut_feats, multi_feat_mut_step

    def create_remaining_mutation_steps(
        self,
        seq_mut_order,
        update_step_start,
    ):
        feat_mutation_steps = [MultiFeatureMutationStep() for i in range(seq_mut_order.obs_seq_mutation.num_mutations - update_step_start)]
        for offset, feat_gen in zip(self.feat_offsets, self.feat_gens):
            mut_steps = feat_gen.create_remaining_mutation_steps(seq_mut_order, update_step_start)
            assert(len(mut_steps) == len(feat_mutation_steps))
            for multi_f, single_f in zip(feat_mutation_steps, mut_steps):
                multi_f.update(single_f, offset)
        return feat_mutation_steps
