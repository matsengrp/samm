import numpy as np
from common import *

class MultinomialSolver:
    @staticmethod
    def solve(obs_data, feat_generator):
        """
        @param obs_data: ObservedSequenceMutations, the observed start and end sequences
        @param feat_generator: FeatureGenerator
        """
        motif_list = feat_generator.get_motif_list()
        num_motifs = len(motif_list)
        # count number of times we see a mutation to a target nucleotide from a motif
        target_mutation_count = np.matrix([[0.] * NUM_NUCLEOTIDES] * num_motifs)
        # count number of times we see a motif
        motif_count = [0] * num_motifs
        for obs_seq in obs_data:
            germline_motifs = feat_generator.create_for_sequence(obs_seq.start_seq)
            for mut_pos, mut_nuc in obs_seq.mutation_pos_dict.iteritems():
                for feature_idx in germline_motifs[mut_pos]:
                    if feature_idx < num_motifs:
                        motif_count[feature_idx] += 1.
                        target_mutation_count[feature_idx, NUCLEOTIDE_DICT[mut_nuc]] += 1.

        proportions = np.matrix([[0.] * NUM_NUCLEOTIDES] * num_motifs)
        for motif_idx, motif_count in enumerate(motif_count):
            if motif_count > 0:
                proportions[motif_idx,] = target_mutation_count[motif_idx,] / motif_count
        return proportions
