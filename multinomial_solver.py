import numpy as np
from common import *

class MultinomialSolver:
    @staticmethod
    def solve(obs_data, feat_generator, theta):
        """
        This model supposes the same hazard rate for different target nucleotides but different probabilities
        of mutating to target nucleotides for each motif. This probability vector can be fitted by finding
        the MLE of the multinomial distribution. So this is just the number of times we observe a mutation to
        a particular target nucleotide out of the number of times we saw this motif mutate.

        When two motifs have similar theta values, we fuse them together and then fit the MLE of their combined
        data.

        TODO: Motifs with theta = 0 will all be fused together - is this a reasonable approach?

        TODO: Right now we just fuse motifs with theta values that are very very similar
        We could be fancy in the future and only fuse motifs if there is a path in the fused
        lasso penalty that connects them.

        @param obs_data: ObservedSequenceMutations, the observed start and end sequences
        @param feat_generator: FeatureGenerator
        @param theta: the fitted theta vector
        @return MLE of a multinomial distribution
        """
        motif_len = feat_generator.motif_len
        half_motif_len = feat_generator.half_motif_len
        motif_list = feat_generator.motif_list

        motif_classes = []
        motif_to_motif_class_dict = dict()
        # Fuse motifs
        already_fused_motifs = set()
        for motif_idx1, m1 in enumerate(motif_list):
            if motif_idx1 not in already_fused_motifs:
                fused_idxs = [motif_idx1]
                # now look for other motifs that it is fused with
                for index_delta, m2 in enumerate(motif_list[motif_idx1 + 1:]):
                    # Must have the same mutating nucleotide
                    if m1[half_motif_len] != m2[half_motif_len]:
                        continue

                    motif_idx2 = motif_idx1 + 1 + index_delta
                    # Cannot be fused already
                    if motif_idx2 in already_fused_motifs:
                        continue

                    # Must have similar theta values
                    if np.isclose(theta[motif_idx1], theta[motif_idx2]):
                        fused_idxs.append(motif_idx2)

                motif_classes.append(fused_idxs)

                motif_class_idx = len(motif_classes) - 1
                for i in fused_idxs:
                    motif_to_motif_class_dict[i] = motif_class_idx
                    already_fused_motifs.add(i)

        num_motif_classes = len(motif_classes)
        # count number of times we see a mutation to a target nucleotide from a motif class
        target_mutation_count = np.matrix([[0.] * NUM_NUCLEOTIDES] * num_motif_classes)
        # count number of times we see a motif class
        motif_class_count = [0] * num_motif_classes
        for obs_seq in obs_data:
            germline_motifs = feat_generator.create_for_sequence(obs_seq.start_seq, obs_seq.left_flank, obs_seq.right_flank)
            for mut_pos, mut_nuc in obs_seq.mutation_pos_dict.iteritems():
                feature_idx = germline_motifs[mut_pos]
                motif_class_idx = motif_to_motif_class_dict[feature_idx]
                motif_class_count[motif_class_idx] += 1.
                target_mutation_count[motif_class_idx, NUCLEOTIDE_DICT[mut_nuc]] += 1.

        # Now get the MLE of the multinomial distributions
        proportions = np.matrix([[0.] * NUM_NUCLEOTIDES] * len(motif_list))
        for i, (motif_class, count) in enumerate(zip(motif_classes, motif_class_count)):
            if count > 0:
                # motifs in the same motif class have the same proportions
                for motif_idx in motif_class:
                    proportions[motif_idx,] = target_mutation_count[i,] / count
        return proportions
