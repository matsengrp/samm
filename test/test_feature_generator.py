import unittest

from feature_generator import SubmotifFeatureGenerator
from models import *

class FeatureGeneratorTestCase(unittest.TestCase):
    def test_update(self):
        feat_generator = SubmotifFeatureGenerator()
        obs_seq_mut = ObservedSequenceMutations(
            start_seq="attacg",
            end_seq="tgcacg"
        )
        ordered_seq_mut = ImputedSequenceMutations(
            obs_seq_mut,
            [0,1,2]
        )

        # Create the base_feat_vec_dicts and base_intermediate_seqs
        # (position 2 mutates last)
        base_feat_vec_dicts, base_intermediate_seqs = feat_generator.create_for_mutation_steps(
            ordered_seq_mut,
        )

        # Compare update to create feature vectors by changing the mutation order by one step
        # (position 2 mutates second)
        ordered_seq_mut1 = ImputedSequenceMutations(
            obs_seq_mut,
            [0,2,1]
        )
        feat_vec_dicts1_update, intermediate_seqs1_update = feat_generator.update_for_mutation_steps(
            ordered_seq_mut1,
            update_steps=[1],
            base_feat_vec_dicts = base_feat_vec_dicts,
            base_intermediate_seqs = base_intermediate_seqs,
        )

        feat_vec_dicts1, intermediate_seqs1 = feat_generator.create_for_mutation_steps(
            ordered_seq_mut1
        )
        self.assertEqual(feat_vec_dicts1_update, feat_vec_dicts1)
        self.assertEqual(intermediate_seqs1_update, intermediate_seqs1)

        # Compare update to create feature vectors by changing the mutation order by another step
        # (position 2 mutates first)
        ordered_seq_mut2 = ImputedSequenceMutations(
            obs_seq_mut,
            [2,0,1]
        )
        feat_vec_dicts2_update, intermediate_seqs2_update = feat_generator.update_for_mutation_steps(
            ordered_seq_mut2,
            update_steps=[0,1],
            base_feat_vec_dicts = feat_vec_dicts1,
            base_intermediate_seqs = intermediate_seqs1,
        )

        feat_vec_dicts2, intermediate_seqs2 = feat_generator.create_for_mutation_steps(
            ordered_seq_mut2
        )
        self.assertEqual(feat_vec_dicts2_update, feat_vec_dicts2)
        self.assertEqual(intermediate_seqs2_update, intermediate_seqs2)
