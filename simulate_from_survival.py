import numpy as np
from common import mutate_string
from common import NUCLEOTIDES
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

    def simulate(self, start_seq, censoring_time):
        """
        @param censoring_time: how long to mutate the sequence for
        @return ending sequence
        """
        mutations = []
        intermediate_seq = start_seq
        unmutated_pos = range(len(start_seq))
        last_mutate_time = 0
        while len(unmutated_pos) > 0:
            # TODO: For speedup, we don't need to recalculate all the features.
            feature_vec_dict = self.feature_generator.create_for_sequence(intermediate_seq, do_feat_vec_pos=unmutated_pos)
            hazard_weights = np.array([
                np.exp(np.sum(self.theta[feature_vec_dict[p]])) for p in unmutated_pos
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
            multinomial_sample = np.random.multinomial(
                n=1,
                pvals=hazard_weights/np.sum(hazard_weights),
            )
            sampled_idx = np.where(multinomial_sample == 1)[0][0]
            mutate_pos = unmutated_pos.pop(sampled_idx)

            # sample the target nucleotide
            nucleotide_original = intermediate_seq[mutate_pos]
            possible_target_nucleotides = NUCLEOTIDES.replace(nucleotide_original, "")
            target_nucleotide_sample = np.random.multinomial(n=1, pvals=np.ones(3)/3)
            nucleotide_target = possible_target_nucleotides[np.where(target_nucleotide_sample == 1)[0][0]]

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
