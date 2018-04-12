import unittest
import time

from adjacent_motif_feature_generator import AdjacentMotifFeatureGenerator
from models import ObservedSequenceMutations, ImputedSequenceMutations

class FeatureGeneratorTestCase(unittest.TestCase):
    def test_create(self):
        motif_len = 3
        feat_generator = AdjacentMotifFeatureGenerator(
            motif_len=motif_len,
            distance_to_right_flank_end=0,
        )
        obs_seq_mut = ObservedSequenceMutations(
            start_seq="caagtatgaatgc",
            end_seq=  "caagcaagatagc",
            motif_len=0,
        )
        feat_matrix = feat_generator.get_base_features(obs_seq_mut)
        obs_seq_mut.set_start_feats(feat_matrix)
        ordered_seq_mut = ImputedSequenceMutations(
            obs_seq_mut,
            sorted(obs_seq_mut.mutation_pos_dict.keys())
        )

        # Create the base_feat_vec_dicts and base_intermediate_seqs
        base_feat_mut_steps = feat_generator.create_for_mutation_steps(ordered_seq_mut)
        self.assertEqual(base_feat_mut_steps[0].mutating_pos_feats, 16 * 0 + 4 * 0 + 1 * 2)
        self.assertEqual(len(base_feat_mut_steps[0].neighbors_feat_new), 0)
        self.assertEqual(len(base_feat_mut_steps[0].neighbors_feat_old), 0)
        self.assertEqual(base_feat_mut_steps[1].mutating_pos_feats, 16 * 2 + 4 * 1 + 1 * 0)
        self.assertEqual(base_feat_mut_steps[1].neighbors_feat_old[3], 16 * 1 + 4 * 0 + 1 * 0)
        self.assertEqual(base_feat_mut_steps[1].neighbors_feat_new[3], 16 * 1 + 4 * 0 + 1 * 0)
