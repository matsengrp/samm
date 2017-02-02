import numpy as np
from common import mutate_string
from common import NUCLEOTIDES
from common import sample_multinomial
from models import *

class SurvivalModelSimulator:
    """
    A simple model that will mutate sequences based on the survival model we've assumed.
    We will suppose that the hazard is constant over time.
    Suppose that when a mutation occurs, it is equally likely to mutate into the
    three other nucleotides (prob=1/3).
    """
    def __init__(self, theta, feature_generator, lambda0):
        """
        @param theta: numpy array of size (p,1)
        @param lambda0: a constant hazard rate
        """
        self.theta = theta
        self.feature_generator = feature_generator
        self.lambda0 = lambda0

    def simulate(self, start_seq, censoring_time, with_replacement=True):
        """
        @param start_seq: string for the original sequence
        @param censoring_time: how long to mutate the sequence for
        @param with_replacement: True = a position can mutate multiple times, False = a position can mutate at most once

        @return FullSequenceMutations, ending sequence and entire history of mutations
        """
        mutations = []
        intermediate_seq = start_seq
        pos_to_mutate = range(len(start_seq))
        last_mutate_time = 0
        while len(pos_to_mutate) > 0:
            # TODO: For speedup, we don't need to recalculate all the features.
            if with_replacement:
                feature_vec_dict = self.feature_generator.create_for_sequence(intermediate_seq)
            else:
                feature_vec_dict = self.feature_generator.create_for_sequence(intermediate_seq, do_feat_vec_pos=pos_to_mutate)
            hazard_weights = np.array([
                np.exp(np.sum(self.theta[feature_vec_dict[p]])) for p in pos_to_mutate
            ])

            # sample the time for the next mutation
            # we do inverse transform sampling - sample from unif and then invert
            unif_sample = np.random.rand(1)
            denom = np.sum(hazard_weights)
            mutate_time = last_mutate_time - 1/self.lambda0 * np.log(1 - unif_sample) / denom
            if censoring_time > mutate_time:
                last_mutate_time = mutate_time
            else:
                break

            # sample the position this mutation occurs in
            # this is a multinomial
            sampled_idx = sample_multinomial(hazard_weights)
            if not with_replacement:
                mutate_pos = pos_to_mutate.pop(sampled_idx)
            else:
                mutate_pos = pos_to_mutate[sampled_idx]

            # sample the target nucleotide
            nucleotide_original = intermediate_seq[mutate_pos]
            possible_target_nucleotides = NUCLEOTIDES.replace(nucleotide_original, "")

            # TODO: Right now this just randomly picks a target nucleotide. Might need to change in the future
            target_nucleotide_idx = sample_multinomial(np.ones(3))
            nucleotide_target = possible_target_nucleotides[target_nucleotide_idx]

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
            ),
            mutations
        )
