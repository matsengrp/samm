import numpy as np

from feature_generator import FeatureGenerator, MultiFeatureMutationStep
from scipy.sparse import hstack

class CombinedFeatureGenerator(FeatureGenerator):
    """
    Class that combines GenericFeatureGenerators.
    """
    def __init__(self, feat_gen_list, model_truncation=None, left_update_region=0, right_update_region=0):
        """
        @param feat_gen_list: list of GenericFeatureGenerators
        @param model_truncation: ModelTruncation
        @param left_update_region: number of positions to consider left of mutating position to update
            features correctly
        @param right_update_region: number of positions to consider right of mutating position to update
            features correctly
        """
        self.feat_gens = feat_gen_list
        self.feats_to_remove = feats_to_remove

        self.left_update_region = left_update_region
        self.right_update_region = right_update_region

        self.update_feats_after_removing(model_truncation)

    def add_base_features_for_list(self, obs_data):
        """
        Calls create_base_features over a list of obs_seq_mutation
        Mutates the obs_data object -- adds starting features as an attribute
        """
        for obs in obs_data:
            self.add_base_features(obs)

    def add_base_features(self, obs_seq_mutation):
        """
        Mutates the `obs_seq_mutation` object -- adds starting features as an attribute
        """
        feat_mats = []
        for offset, feat_gen in zip(self.feat_offsets, self.feat_gens):
            feat_mat = feat_gen.get_base_features(obs_seq_mutation)
            feat_mats.append(feat_mat)
        full_feat_mat = hstack(feat_mats, format="csr")
        obs_seq_mutation.set_start_feats(full_feat_mat)

    def create_for_mutation_steps(self, seq_mut_order):
        feat_mutation_steps = [MultiFeatureMutationStep() for i in range(seq_mut_order.obs_seq_mutation.num_mutations)]
        for offset, feat_gen in zip(self.feat_offsets, self.feat_gens):
            mut_steps = feat_gen.create_for_mutation_steps(seq_mut_order, self.left_update_region, self.right_update_region)
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
            mut_pos_feat, mut_step = feat_gen.get_shuffled_mutation_steps_delta(
                seq_mut_order,
                update_step,
                flanked_seq,
                already_mutated_pos,
                self.left_update_region,
                self.right_update_region,
            )
            if mut_pos_feat is not None:
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
            mut_steps = feat_gen.create_remaining_mutation_steps(seq_mut_order, update_step_start, self.left_update_region, self.right_update_region)
            assert(len(mut_steps) == len(feat_mutation_steps))
            for multi_f, single_f in zip(feat_mutation_steps, mut_steps):
                multi_f.update(single_f, offset)
        return feat_mutation_steps

    def get_possible_motifs_to_targets(self, mask_shape):
        """
        For a per-target model, this is the matrix with elements corresponding to which nucleotides certain features can mutate to.
        Default is any feature can mutate to any nucleotide

        @param mask_shape: shape of theta mask; either p-by-1 or p-by-(NUM_NUCLEOTIDES+1)
        @return a boolean matrix with possible mutations as True, impossible mutations as False
        """
        return np.ones(mask_shape, dtype=bool)

    def update_feats_after_removing(self, model_truncation):
        """
        Updates feature generator properties after removing features

        @param model_truncation: ModelTruncation
        """
        self.model_truncation = model_truncation
        self.feats_to_remove = model_truncation.feats_to_remove if model_truncation is not None else []

        # Create list of feature generators for different motif lengths and different flank lengths
        old_feat_gens = self.feat_gens
        self.feat_gens = []
        self.feature_info_list = []
        for feat_gen in old_feat_gens:
            feat_gen.update_feats_after_removing(self.model_truncation)
            self.feat_gens.append(feat_gen)
            self.feature_info_list += feat_gen.feature_info_list

        feat_offsets = [feat_gen.feature_vec_len for feat_gen in self.feat_gens]
        self.feat_offsets = np.cumsum([0] + feat_offsets)[:-1]
        self.num_feat_gens = len(self.feat_gens)
        self.feature_vec_len = np.sum(feat_offsets)

    def create_for_sequence(self, seq_str, left_flank, right_flank, do_feat_vec_pos=None):
        if do_feat_vec_pos is None:
            do_feat_vec_pos = range(len(seq_str))
        feat_vec_dict = {pos:[] for pos in do_feat_vec_pos}
        for offset, feat_gen in zip(self.feat_offsets, self.feat_gens):
            f_dict = feat_gen.create_for_sequence(seq_str, left_flank, right_flank, do_feat_vec_pos)
            for pos in do_feat_vec_pos:
                feat = f_dict[pos]
                if feat is not None:
                    feat_vec_dict[pos].append(feat + offset)
        return feat_vec_dict

    def print_label_from_idx(self, idx):
        label = ""
        for offset, feat_gen in zip(self.feat_offsets, self.feat_gens):
            if idx < offset + feat_gen.feature_vec_len:
                label = feat_gen.print_label_from_info(self.feature_info_list[idx])
                break
        return label
