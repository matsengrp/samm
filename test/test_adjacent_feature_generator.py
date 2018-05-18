import unittest
import time

from motif_feature_generator import MotifFeatureGenerator
from models import ObservedSequenceMutations, ImputedSequenceMutations

class FeatureGeneratorTestCase(unittest.TestCase):
    def test_create(self):
        motif_len = 3
        distance_to_start_of_motif = -3
        left_update = -distance_to_start_of_motif
        right_update = 0
        feat_generator = MotifFeatureGenerator(
            motif_len=motif_len,
            distance_to_start_of_motif=distance_to_start_of_motif,
        )
        obs_seq_mut = ObservedSequenceMutations(
            start_seq="caagtatgaatgc",
            end_seq=  "caagcaagatagc",
            motif_len=3,
            left_flank_len=-distance_to_start_of_motif,
            right_flank_len=0,
        )
        feat_matrix = feat_generator.get_base_features(obs_seq_mut)
        obs_seq_mut.set_start_feats(feat_matrix)
        ordered_seq_mut = ImputedSequenceMutations(
            obs_seq_mut,
            sorted(obs_seq_mut.mutation_pos_dict.keys())
        )

        # Create the base_feat_vec_dicts and base_intermediate_seqs
        base_feat_mut_steps = feat_generator.create_for_mutation_steps(ordered_seq_mut, left_update, right_update)
        self.assertEqual(base_feat_mut_steps[0].mutating_pos_feats, 16 * 0 + 4 * 0 + 1 * 2)
        self.assertEqual(len(base_feat_mut_steps[0].neighbors_feat_new), 0)
        self.assertEqual(len(base_feat_mut_steps[0].neighbors_feat_old), 0)
        # Neighbor feats are indexed by ignoring flanks---took me a while to fix this...
        self.assertEqual(base_feat_mut_steps[1].mutating_pos_feats, 16 * 2 + 4 * 1 + 1 * 0)
        self.assertEqual(base_feat_mut_steps[1].neighbors_feat_old[0], 16 * 1 + 4 * 0 + 1 * 0)
        self.assertEqual(base_feat_mut_steps[1].neighbors_feat_new[0], 16 * 1 + 4 * 0 + 1 * 0)
