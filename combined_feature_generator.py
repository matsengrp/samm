import numpy as np

from feature_generator import FeatureGenerator, MultiFeatureMutationStep
from scipy.sparse import hstack

class CombinedFeatureGenerator(FeatureGenerator):
    """
    A generic way to combine feature generators.
    A hierarchical model is a type of this.
    """
    def __init__(self, feat_gen_list):
        """
        @param feat_gen_list: list of feature generators
        """
        self.feat_gens = feat_gen_list

        feat_offsets = [feat_gen.feature_vec_len for feat_gen in self.feat_gens]
        self.feat_offsets = np.cumsum([0] + feat_offsets)[:-1]
        self.num_feat_gens = len(self.feat_gens)
        self.feature_vec_len = np.sum(feat_offsets)
        self.feature_label_list = []
        for f in self.feat_gens:
            self.feature_label_list += f.feature_label_list

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
            feat_mat = feat_gen.get_base_features(obs_seq_mutation, feat_gen.feature_vec_len)
            feat_mats.append(feat_mat)
        full_feat_mat = hstack(feat_mats, format="csr")
        obs_seq_mutation.set_start_feats(full_feat_mat)

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
            mut_pos_feat, mut_step = feat_gen.get_shuffled_mutation_steps_delta(
                seq_mut_order,
                update_step,
                flanked_seq,
                already_mutated_pos,
            )
            if mut_pos_feat is not None:
                first_mut_feats += [feat + offset for feat in mut_pos_feat]
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

    def get_possible_motifs_to_targets(self, mask_shape):
        """
        @return a boolean matrix with possible mutations as True, impossible mutations as False
        """
        # default to no per-target mask?
        return np.ones(mask_shape, dtype=bool)

    def _update_feature_generator_after_removing(self, model_masks):
        """
        so we don't have to create a whole new feature vector
        """
        # Create list of feature generators for different motif lengths and different flank lengths
        old_feat_gens = self.feat_gens
        self.feat_gens = []
        for feat_gen in old_feat_gens:
            feat_gen._update_motifs_after_removing(model_masks.feats_to_remove)
            self.feat_gens.append(feat_gen)

        feat_offsets = [feat_gen.feature_vec_len for feat_gen in self.feat_gens]
        self.feat_offsets = np.cumsum([0] + feat_offsets)[:-1]
        self.feature_vec_len = np.sum(feat_offsets)
