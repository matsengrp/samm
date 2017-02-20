import unittest
import time

from submotif_feature_generator import SubmotifFeatureGenerator
from submotif_feature_generator_fast import SubmotifFastFeatureGenerator
from models import *
from common import *

class FeatureGeneratorTestCase(unittest.TestCase):
    def test_time(self):
        """
        Just a test to see how fast things are running
        """
        np.random.seed(0)

        motif_len = 3
        seq_length = 400
        mut_per_length = 10

        feat_generator = SubmotifFastFeatureGenerator(motif_len=motif_len)

        start_seq = get_random_dna_seq(seq_length)
        # Mutate a 10th of the sequence
        end_seq = list(start_seq)
        for i in range(motif_len/2, seq_length, mut_per_length):
            if NUCLEOTIDE_DICT[end_seq[i]] == 0:
                end_seq[i] = "t"
            else:
                end_seq[i] = NUCLEOTIDES[NUCLEOTIDE_DICT[end_seq[i]] - 1]
        end_seq = "".join(end_seq)

        obs_seq_mutation = ObservedSequenceMutations(start_seq, end_seq, motif_len)

        st_time = time.time()
        obs_seq_mutation = feat_generator.create_base_features(obs_seq_mutation)
        print "time", time.time() - st_time

        my_order = obs_seq_mutation.mutation_pos_dict.keys()
        seq_mut_order = ImputedSequenceMutations(
            obs_seq_mutation,
            my_order,
        )
        st_time = time.time()
        mutation_steps = feat_generator.create_for_mutation_steps(seq_mut_order)
        print "time", time.time() - st_time

    def test_update(self):
        feat_generator = SubmotifFeatureGenerator(motif_len=3)
        obs_seq_mut = feat_generator.create_base_features(
            ObservedSequenceMutations(
                start_seq="aattacgc",
                end_seq="atgcacgc",
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
