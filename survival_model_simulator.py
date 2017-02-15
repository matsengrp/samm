import numpy as np
from common import mutate_string
from common import NUCLEOTIDES
from common import sample_multinomial
from models import *

class SurvivalModelSimulator:
    """
    A simple model that will mutate sequences based on the survival model we've assumed.
    We will suppose that the hazard is constant over time.
    """
    def __init__(self, thetas, feature_generator, lambda0):
        """
        @param thetas: numpy array of size (p,4) where p is number of motifs
                       In each row, theta_row[i] represents the rate to mutate to the i-th nucleotide in the string NUCLEOTIDES
                       Note that the theta values for target nucleotides that have the SAME center nucleotide as the motif
                       should be completey ignored. (They should technically be negative infinity)
        @param lambda0: a constant hazard rate
        """
        self.thetas = thetas
        self.feature_generator = feature_generator
        self.lambda0 = lambda0

    def simulate(self, start_seq, censoring_time, with_replacement=False):
        """
        @param start_seq: string for the original sequence
        @param censoring_time: how long to mutate the sequence for
        @param with_replacement: True = a position can mutate multiple times, False = a position can mutate at most once

        @return FullSequenceMutations, ending sequence and entire history of mutations
        """
        mutations = []
        intermediate_seq = start_seq
        pos_to_mutate = set(range(len(start_seq)))
        last_mutate_time = 0
        while len(pos_to_mutate) > 0:
            # TODO: For speedup, we don't need to recalculate all the features.
            if with_replacement:
                feature_vec_dict = self.feature_generator.create_for_sequence(intermediate_seq)
            else:
                feature_vec_dict = self.feature_generator.create_for_sequence(intermediate_seq, do_feat_vec_pos=pos_to_mutate)

            # Fill in all the hazard weights -- only fill in for the positions eligible for mutation
            # and only fill in target nucleotides that are possible to mutate into.
            # Note: shmulate removes all stop codons too, which we will not be doing here.
            hazard_weights = []
            mutate_positions = []
            target_nucleotides = []
            for p in pos_to_mutate:
                motif_idx = feature_vec_dict[p]
                nucleotide_to_mutate = intermediate_seq[p]
                for theta_idx, target_nucleotide in enumerate(NUCLEOTIDES):
                    if target_nucleotide != nucleotide_to_mutate:
                        hazard_weights.append(np.exp(np.sum(self.thetas[motif_idx, theta_idx])))
                        target_nucleotides.append(target_nucleotide)
                        mutate_positions.append(p)

            # sample the time for the next mutation
            # we do inverse transform sampling - sample from unif and then invert
            unif_sample = np.random.rand(1)
            denom = np.sum(hazard_weights)
            mutate_time = last_mutate_time - 1/self.lambda0 * np.log(1 - unif_sample) / denom
            if censoring_time > mutate_time:
                last_mutate_time = mutate_time
            else:
                break

            # sample mutation position and target nucleotide
            # this is a multinomial
            sampled_idx = sample_multinomial(hazard_weights)
            mutate_pos = mutate_positions[sampled_idx]
            nucleotide_target = target_nucleotides[sampled_idx]
            if not with_replacement:
                pos_to_mutate.remove(mutate_pos)

            mutations.append(MutationEvent(
                mutate_time,
                mutate_pos,
                nucleotide_target,
            ))

            intermediate_seq = mutate_string(intermediate_seq, mutate_pos, nucleotide_target)
        return FullSequenceMutations(
            ObservedSequenceMutations(
                start_seq,
                intermediate_seq,
                self.feature_generator.motif_len,
            ),
            mutations
        )
