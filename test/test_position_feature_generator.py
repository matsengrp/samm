import unittest
import time

from position_feature_generator import PositionFeatureGenerator
from models import ObservedSequenceMutations, ImputedSequenceMutations
from common import process_degenerates_and_impute_nucleotides

class FeatureGeneratorTestCase(unittest.TestCase):
    def test_create(self):
        motif_len = 3
        distance_to_start_of_motif = -1
        left_update = 1
        right_update = 1

        start_seq = "nnncaagtannnnnntgaatgc"
        end_seq =   "nnncaagcannnnnnagatagc"

        # defaults to each position is its own feature---useful for testing
        feat_generator = PositionFeatureGenerator()

        start_seq_proc, end_seq_proc, collapse_list = process_degenerates_and_impute_nucleotides(
            start_seq,
            end_seq,
            motif_len,
        )
        obs_seq_mut = ObservedSequenceMutations(
            start_seq=start_seq_proc,
            end_seq=end_seq_proc,
            motif_len=motif_len,
            left_flank_len=-distance_to_start_of_motif,
            right_flank_len=-distance_to_start_of_motif,
            collapse_list=collapse_list,
        )
        feat_matrix = feat_generator.get_base_features(obs_seq_mut)
        obs_seq_mut.set_start_feats(feat_matrix)
        ordered_seq_mut = ImputedSequenceMutations(
            obs_seq_mut,
            sorted(obs_seq_mut.mutation_pos_dict.keys())
        )

        # Create the base_feat_vec_dicts and base_intermediate_seqs
        base_feat_mut_steps = feat_generator.create_for_mutation_steps(ordered_seq_mut, left_update, right_update)
        self.assertEqual(base_feat_mut_steps[0].mutating_pos_feats, 7)
        self.assertEqual(len(base_feat_mut_steps[0].neighbors_feat_new), 0)
        self.assertEqual(len(base_feat_mut_steps[0].neighbors_feat_old), 0)
        self.assertEqual(base_feat_mut_steps[1].mutating_pos_feats, 15)
        self.assertEqual(base_feat_mut_steps[2].mutating_pos_feats, 18)
        self.assertEqual(base_feat_mut_steps[3].mutating_pos_feats, 19)

