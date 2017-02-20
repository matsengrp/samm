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
        print "create_base_features time", time.time() - st_time

        my_order = obs_seq_mutation.mutation_pos_dict.keys()
        seq_mut_order = ImputedSequenceMutations(
            obs_seq_mutation,
            my_order,
        )
        st_time = time.time()
        mutation_steps = feat_generator.create_for_mutation_steps(seq_mut_order)
        print "create_for_mutation_steps time", time.time() - st_time

    def test_update(self):
        feat_generator = SubmotifFastFeatureGenerator(motif_len=3)
        obs_seq_mut = feat_generator.create_base_features(
            ObservedSequenceMutations(
                start_seq="aattatgaatgc",
                end_seq=  "atgcaagatagc",
                motif_len=3,
            )
        )
        self.assertEqual(obs_seq_mut.feat_dict_start[0], 3)
        self.assertEqual(obs_seq_mut.feat_dict_start[1], 4 * 3 + 3)
        ordered_seq_mut = ImputedSequenceMutations(
            obs_seq_mut,
            obs_seq_mut.mutation_pos_dict.keys()
        )

        # Create the base_feat_vec_dicts and base_intermediate_seqs
        # (position 2 mutates last)
        base_feat_mut_steps = feat_generator.create_for_mutation_steps(ordered_seq_mut)
        self.assertEqual(base_feat_mut_steps[0].mutating_pos_feat, 3)
        self.assertEqual(len(base_feat_mut_steps[0].neighbors_feat_new), 0)
        self.assertEqual(len(base_feat_mut_steps[0].neighbors_feat_old), 0)
        self.assertEqual(base_feat_mut_steps[1].mutating_pos_feat, 16 * 3 + 3 * 4 + 3)
        self.assertEqual(base_feat_mut_steps[1].neighbors_feat_new.keys(), [1])
        self.assertEqual(base_feat_mut_steps[1].neighbors_feat_old.keys(), [1])
        self.assertEqual(set(base_feat_mut_steps[4].neighbors_feat_new.keys()), set([3,5]))
        self.assertEqual(set(base_feat_mut_steps[4].neighbors_feat_old.keys()), set([3,5]))

        # Compare update to create feature vectors by changing the mutation order by one step
        # (position 2 mutates second)
        new_order = obs_seq_mut.mutation_pos_dict.keys()
        new_order = new_order[0:-2] + [new_order[-1], new_order[-2]]
        ordered_seq_mut1 = ImputedSequenceMutations(
            obs_seq_mut,
            new_order
        )

        feat_mut_steps1 = feat_generator.create_for_mutation_steps(ordered_seq_mut1)
        first_mutation_feat, second_mut_step = feat_generator.update_for_mutation_steps(
            ordered_seq_mut1,
            update_steps=[obs_seq_mut.num_mutations - 2, obs_seq_mut.num_mutations - 1],
        )
        self.assertEqual(first_mutation_feat, 14)
        self.assertEqual(feat_mut_steps1[-2].mutating_pos_feat, 14)
        self.assertEqual(second_mut_step.mutating_pos_feat, 0)
        self.assertEqual(feat_mut_steps1[-1].mutating_pos_feat, 0)

        # Compare update to create feature vectors by changing the mutation order by another step
        new_order = new_order[0:-2] + [new_order[-1], new_order[-2]]
        ordered_seq_mut2 = ImputedSequenceMutations(
            obs_seq_mut,
            new_order
        )

        feat_mut_steps2 = feat_generator.create_for_mutation_steps(ordered_seq_mut2)
        first_mutation_feat2, second_mut_step2 = feat_generator.update_for_mutation_steps(
            ordered_seq_mut2,
            update_steps=[obs_seq_mut.num_mutations - 3, obs_seq_mut.num_mutations - 2],
        )
        self.assertEqual(first_mutation_feat2, 14)
        self.assertEqual(second_mut_step2.mutating_pos_feat, 3)
        self.assertEqual(second_mut_step2.neighbors_feat_old, feat_mut_steps2[-2].neighbors_feat_old)
        self.assertEqual(second_mut_step2.neighbors_feat_new, feat_mut_steps2[-2].neighbors_feat_new)
