import unittest
import time

from submotif_feature_generator import SubmotifFeatureGenerator
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

        feat_generator = SubmotifFeatureGenerator(motif_len=motif_len)

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

    def test_create(self):
        motif_len = 3
        feat_generator = SubmotifFeatureGenerator(motif_len=motif_len)
        obs_seq_mut = feat_generator.create_base_features(
            ObservedSequenceMutations(
                start_seq="aattatgaatgc",
                end_seq=  "atgcaagatagc",
                motif_len=3,
            )
        )
        ordered_seq_mut = ImputedSequenceMutations(
            obs_seq_mut,
            obs_seq_mut.mutation_pos_dict.keys()
        )

        # Create the base_feat_vec_dicts and base_intermediate_seqs
        base_feat_mut_steps = feat_generator.create_for_mutation_steps(ordered_seq_mut)
        self.assertEqual(base_feat_mut_steps[0].mutating_pos_feats, 16 * 0 + 4 * 0 + 1 * 3)
        self.assertEqual(len(base_feat_mut_steps[0].neighbors_feat_new), 0)
        self.assertEqual(len(base_feat_mut_steps[0].neighbors_feat_old), 0)
        self.assertEqual(base_feat_mut_steps[1].mutating_pos_feats, 16 * 3 + 4 * 3 + 1 * 3)
        self.assertEqual(base_feat_mut_steps[1].neighbors_feat_old[1], 16 * 0 + 4 * 3 + 1 * 3)
        self.assertEqual(base_feat_mut_steps[1].neighbors_feat_new[1], 16 * 3 + 4 * 3 + 1 * 3)
        self.assertEqual(base_feat_mut_steps[1].neighbors_feat_new.keys(), [1])
        self.assertEqual(base_feat_mut_steps[1].neighbors_feat_old.keys(), [1])
        self.assertEqual(set(base_feat_mut_steps[4].neighbors_feat_new.keys()), set([3,5]))
        self.assertEqual(set(base_feat_mut_steps[4].neighbors_feat_old.keys()), set([3,5]))

    def test_create_upstream(self):
        motif_len = 3
        left_motif_flank_len = 0
        feat_generator = SubmotifFeatureGenerator(motif_len=motif_len,
                left_motif_flank_len=left_motif_flank_len,
                left_update_region=0,
                right_update_region=2)
        obs_seq_mut = feat_generator.create_base_features(
            ObservedSequenceMutations(
                start_seq="aattatgaatgc",
                end_seq=  "atgcaagatagc",
                motif_len=3,
                left_flank_len=left_motif_flank_len,
                right_flank_len=motif_len - 1 - left_motif_flank_len,
            )
        )
        ordered_seq_mut = ImputedSequenceMutations(
            obs_seq_mut,
            obs_seq_mut.mutation_pos_dict.keys()
        )

        # Create the base_feat_vec_dicts and base_intermediate_seqs
        base_feat_mut_steps = feat_generator.create_for_mutation_steps(ordered_seq_mut)
        self.assertEqual(base_feat_mut_steps[0].mutating_pos_feats, 16 * 0 + 4 * 3  + 1 * 3)
        self.assertEqual(len(base_feat_mut_steps[0].neighbors_feat_new), 0)
        self.assertEqual(len(base_feat_mut_steps[0].neighbors_feat_old), 0)
        self.assertEqual(base_feat_mut_steps[1].mutating_pos_feats, 16 * 3 + 4 * 3 + 1 * 0)
        self.assertEqual(base_feat_mut_steps[1].neighbors_feat_old[2], 16 * 3 + 4 * 3 + 1 * 0)
        self.assertEqual(base_feat_mut_steps[1].neighbors_feat_new[2], 16 * 3 + 4 * 3 + 1 * 0)
        self.assertEqual(base_feat_mut_steps[1].neighbors_feat_old[3], 16 * 3 + 4 * 0 + 1 * 3)
        self.assertEqual(base_feat_mut_steps[1].neighbors_feat_new[3], 16 * 3 + 4 * 0 + 1 * 3)
        self.assertEqual(base_feat_mut_steps[4].neighbors_feat_new.keys(), [6,7])
        self.assertEqual(base_feat_mut_steps[4].neighbors_feat_old.keys(), [6,7])

    def test_create_downstream(self):
        motif_len = 3
        left_motif_flank_len = 2
        feat_generator = SubmotifFeatureGenerator(motif_len=motif_len,
                left_motif_flank_len=left_motif_flank_len,
                left_update_region=2,
                right_update_region=0)
        obs_seq_mut = feat_generator.create_base_features(
            ObservedSequenceMutations(
                start_seq="aaattatgaatgc",
                end_seq=  "aatgcaagatagc",
                motif_len=3,
                left_flank_len=left_motif_flank_len,
                right_flank_len=motif_len - 1 - left_motif_flank_len,
            )
        )
        ordered_seq_mut = ImputedSequenceMutations(
            obs_seq_mut,
            obs_seq_mut.mutation_pos_dict.keys()
        )

        # Create the base_feat_vec_dicts and base_intermediate_seqs
        base_feat_mut_steps = feat_generator.create_for_mutation_steps(ordered_seq_mut)
        self.assertEqual(base_feat_mut_steps[0].mutating_pos_feats, 16 * 0 + 4 * 0  + 1 * 0)
        self.assertEqual(len(base_feat_mut_steps[0].neighbors_feat_new), 0)
        self.assertEqual(len(base_feat_mut_steps[0].neighbors_feat_old), 0)
        self.assertEqual(base_feat_mut_steps[1].mutating_pos_feats, 16 * 0 + 4 * 3 + 1 * 3)
        self.assertEqual(len(base_feat_mut_steps[1].neighbors_feat_new), 0)
        self.assertEqual(len(base_feat_mut_steps[1].neighbors_feat_old), 0)
        self.assertEqual(base_feat_mut_steps[4].neighbors_feat_old.keys(), [3])
        self.assertEqual(base_feat_mut_steps[4].neighbors_feat_new.keys(), [3])

    def test_create_all(self):
        motif_len = 3
        left_flank_lens = [1, 2]
        left_motif_flank_len = 1
        feat_generator1 = SubmotifFeatureGenerator(motif_len=motif_len,
                left_motif_flank_len=left_motif_flank_len,
                hier_offset=max(left_flank_lens) - left_motif_flank_len,
                left_update_region=max(left_flank_lens),
                right_update_region=motif_len - 1 - min(left_flank_lens))
        obs_seq_mut1 = feat_generator1.create_base_features(
            ObservedSequenceMutations(
                start_seq="aaattatgaatgc",
                end_seq=  "aatgcaagatagc",
                motif_len=3,
                left_flank_len=max(left_flank_lens),
                right_flank_len=motif_len - 1 - min(left_flank_lens),
            )
        )
        ordered_seq_mut1 = ImputedSequenceMutations(
            obs_seq_mut1,
            obs_seq_mut1.mutation_pos_dict.keys()
        )

        left_motif_flank_len = 2
        feat_generator2 = SubmotifFeatureGenerator(motif_len=motif_len,
                left_motif_flank_len=left_motif_flank_len,
                hier_offset=max(left_flank_lens) - left_motif_flank_len,
                left_update_region=max(left_flank_lens),
                right_update_region=motif_len - 1 - min(left_flank_lens))
        obs_seq_mut2 = feat_generator2.create_base_features(
            ObservedSequenceMutations(
                start_seq="aaattatgaatgc",
                end_seq=  "aatgcaagatagc",
                motif_len=3,
                left_flank_len=max(left_flank_lens),
                right_flank_len=motif_len - 1 - min(left_flank_lens),
            )
        )
        ordered_seq_mut2 = ImputedSequenceMutations(
            obs_seq_mut2,
            obs_seq_mut2.mutation_pos_dict.keys()
        )

        # Create the base_feat_vec_dicts and base_intermediate_seqs
        base_feat_mut_steps1 = feat_generator1.create_for_mutation_steps(ordered_seq_mut1)
        base_feat_mut_steps2 = feat_generator2.create_for_mutation_steps(ordered_seq_mut2)
        self.assertEqual(base_feat_mut_steps1[0].mutating_pos_feats, 16 * 0 + 4 * 0  + 1 * 3)
        self.assertEqual(base_feat_mut_steps2[0].mutating_pos_feats, 16 * 0 + 4 * 0  + 1 * 0)
        self.assertEqual(base_feat_mut_steps1[1].neighbors_feat_old[1], 16 * 0 + 4 * 3  + 1 * 3)
        self.assertEqual(base_feat_mut_steps1[1].neighbors_feat_new[1], 16 * 3 + 4 * 3  + 1 * 3)

    def test_update(self):
        motif_len = 3
        feat_generator = SubmotifFeatureGenerator(motif_len=motif_len)
        obs_seq_mut = feat_generator.create_base_features(
            ObservedSequenceMutations(
                start_seq="aattatgaatgc",
                end_seq=  "atgcaagatagc",
                motif_len=3,
            )
        )

        # Compare update to create feature vectors by changing the mutation order by one step
        # Shuffle last two positions
        new_order = obs_seq_mut.mutation_pos_dict.keys()
        new_order = new_order[0:-2] + [new_order[-1], new_order[-2]]
        ordered_seq_mut1 = ImputedSequenceMutations(
            obs_seq_mut,
            new_order
        )
        # Revert the sequence back two steps
        intermediate_seq = obs_seq_mut.end_seq
        intermediate_seq = (
            intermediate_seq[:new_order[-2]]
            + obs_seq_mut.start_seq[new_order[-2]]
            + intermediate_seq[new_order[-2] + 1:]
        )
        intermediate_seq = (
            intermediate_seq[:new_order[-1]]
            + obs_seq_mut.start_seq[new_order[-1]]
            + intermediate_seq[new_order[-1] + 1:]
        )
        flanked_seq = (
            obs_seq_mut.left_flank
            + intermediate_seq
            + obs_seq_mut.right_flank
        )
        # create features - the slow version
        feat_mut_steps1 = feat_generator.create_for_mutation_steps(ordered_seq_mut1)
        # get the feature delta - the fast version
        first_mutation_feat, second_mut_step = feat_generator.get_shuffled_mutation_steps_delta(
            ordered_seq_mut1,
            update_step=obs_seq_mut.num_mutations - 2,
            flanked_seq=flanked_seq,
            already_mutated_pos=set(new_order[:obs_seq_mut.num_mutations - 2]),
        )
        self.assertEqual(first_mutation_feat, 14)
        self.assertEqual(feat_mut_steps1[-2].mutating_pos_feats, 14)
        self.assertEqual(second_mut_step.mutating_pos_feats, 0)
        self.assertEqual(feat_mut_steps1[-1].mutating_pos_feats, 0)

        # Compare update to create feature vectors by changing the mutation order by another step
        # Shuffle second to last with the third to last mutation positions
        flanked_seq = (
            flanked_seq[:motif_len/2 + new_order[-3]]
            + obs_seq_mut.start_seq[new_order[-3]]
            + flanked_seq[motif_len/2 + new_order[-3] + 1:]
        )
        new_order = new_order[0:-3] + [new_order[-2], new_order[-3], new_order[-1]]
        ordered_seq_mut2 = ImputedSequenceMutations(
            obs_seq_mut,
            new_order
        )

        # create features - the slow version
        feat_mut_steps2 = feat_generator.create_for_mutation_steps(ordered_seq_mut2)
        # get the feature delta - the fast version
        first_mutation_feat2, second_mut_step2 = feat_generator.get_shuffled_mutation_steps_delta(
            ordered_seq_mut2,
            update_step=obs_seq_mut.num_mutations - 3,
            flanked_seq=flanked_seq,
            already_mutated_pos=set(new_order[:obs_seq_mut.num_mutations - 3]),
        )
        self.assertEqual(first_mutation_feat2, 14)
        self.assertEqual(second_mut_step2.mutating_pos_feats, 14)
        self.assertEqual(second_mut_step2.neighbors_feat_old, {9: 57, 7: 3})
        self.assertEqual(second_mut_step2.neighbors_feat_new, {9: 9, 7: 0})
        self.assertEqual(second_mut_step2.neighbors_feat_old, feat_mut_steps2[-2].neighbors_feat_old)
        self.assertEqual(second_mut_step2.neighbors_feat_new, feat_mut_steps2[-2].neighbors_feat_new)
