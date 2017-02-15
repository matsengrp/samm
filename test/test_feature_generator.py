import unittest

from submotif_feature_generator import SubmotifFeatureGenerator
from models import *

class FeatureGeneratorTestCase(unittest.TestCase):
    def test_update(self):
        feat_generator = SubmotifFeatureGenerator()
        obs_seq_mut = feat_generator.create_base_features(
            ObservedSequenceMutations(
                start_seq="attacg",
                end_seq="tgcacg"
                motif_len=3,
            )
        )

        ordered_seq_mut = ImputedSequenceMutations(
            obs_seq_mut,
            [0,1,2]
        )

        # Create the base_feat_vec_dicts and base_intermediate_seqs
        # (position 2 mutates last)
        base_feat_mut_steps, base_intermediate_seqs = feat_generator.create_for_mutation_steps(
            ordered_seq_mut,
        )

        # Compare update to create feature vectors by changing the mutation order by one step
        # (position 2 mutates second)
        ordered_seq_mut1 = ImputedSequenceMutations(
            obs_seq_mut,
            [0,2,1]
        )
        feat_mut_steps1_update, _ = feat_generator.update_for_mutation_steps(
            ordered_seq_mut1,
            update_steps=[1],
            base_feat_mutation_steps = base_feat_mut_steps,
        )

        feat_mut_steps1, _ = feat_generator.create_for_mutation_steps(
            ordered_seq_mut1
        )
        self.assertEqual(feat_mut_steps1.intermediate_seqs, feat_mut_steps1_update.intermediate_seqs)
        self.assertEqual(feat_mut_steps1.feature_vec_dicts, feat_mut_steps1_update.feature_vec_dicts)

        # Compare update to create feature vectors by changing the mutation order by another step
        # (position 2 mutates first)
        ordered_seq_mut2 = ImputedSequenceMutations(
            obs_seq_mut,
            [2,0,1]
        )
        feat_mut_steps2_update, _ = feat_generator.update_for_mutation_steps(
            ordered_seq_mut2,
            update_steps=[0,1],
            base_feat_mutation_steps = feat_mut_steps1,
        )

        feat_mut_steps2, _ = feat_generator.create_for_mutation_steps(
            ordered_seq_mut2
        )
        self.assertEqual(feat_mut_steps2.intermediate_seqs, feat_mut_steps2_update.intermediate_seqs)
        self.assertEqual(feat_mut_steps2.feature_vec_dicts, feat_mut_steps2_update.feature_vec_dicts)
