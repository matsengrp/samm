import numpy as np
from common import mutate_string
from common import NUCLEOTIDES, NUM_NUCLEOTIDES
from common import sample_multinomial, process_degenerates_and_impute_nucleotides
from models import *

class SurvivalModelSimulator:
    """
    A simple model that will mutate sequences based on the survival model we've assumed.
    We will suppose that the hazard is constant over time.
    """
    def simulate(self, start_seq, left_flank=None, right_flank=None, censoring_time=None, percent_mutated=None, with_replacement=False):
        """
        @param start_seq: string for the original sequence; includes flanks unless they are provided by left_flank/right_flank
        @param left_flank: the left flank
        @param right_flank: the right flank
        @param censoring_time: how long to mutate the sequence for
        @param percent_mutated: percent of sequence to mutated
        @param with_replacement: True = a position can mutate multiple times, False = a position can mutate at most once

        @return FullSequenceMutations, ending sequence and entire history of mutations
        """
        mutations = []

        if left_flank is None and right_flank is None:
            left_flank = start_seq[:self.feature_generator.max_left_motif_flank_len]
            right_flank = start_seq[len(start_seq) - self.feature_generator.max_right_motif_flank_len:]
            start_seq = start_seq[self.feature_generator.max_left_motif_flank_len:len(start_seq) - self.feature_generator.max_right_motif_flank_len]

        pos_to_mutate = set(range(len(start_seq)))
        intermediate_seq = start_seq
        last_mutate_time = 0
        while len(pos_to_mutate) > 0:
            # TODO: For speedup, we don't need to recalculate all the features.
            if with_replacement:
                feature_vec_dict = self.feature_generator.create_for_sequence(intermediate_seq, left_flank, right_flank)
            else:
                feature_vec_dict = self.feature_generator.create_for_sequence(intermediate_seq, left_flank, right_flank, do_feat_vec_pos=pos_to_mutate)

            mutate_time_delta, mutate_pos, nucleotide_target = self._sample_mutation(feature_vec_dict, intermediate_seq, pos_to_mutate)
            mutate_time = last_mutate_time + mutate_time_delta

            if censoring_time is not None and censoring_time < mutate_time:
                break
            elif percent_mutated is not None and len(mutations) > percent_mutated * len(start_seq):
                break

            last_mutate_time = mutate_time

            if not with_replacement:
                pos_to_mutate.remove(mutate_pos)

            mutations.append(MutationEvent(
                mutate_time,
                mutate_pos,
                nucleotide_target,
            ))

            intermediate_seq = mutate_string(intermediate_seq, mutate_pos, nucleotide_target)
        return FullSequenceMutations(
            start_seq,
            intermediate_seq,
            left_flank,
            right_flank,
            mutations,
        )

    def simulate_dataset_from_observed(self, obs_data, with_replacement=False, motif_len=5):
        """
        Simulates a dataset with similar germlines and mutation positions/rates as an observed dataset

        @param obs_data: data to mimic in simulation
        @param with_replacement: True = a position can mutate multiple times, False = a position can mutate at most once
        @param motif_len: length of motif, for data processing

        @return list of ObservedSequenceMutations
        """
        simulated_data = []
        for idx, obs_seq_mutation in enumerate(obs_data):
            pos_to_mutate = obs_seq_mutation.mutation_pos_dict.keys()
            sample = self.simulate(start_seq=obs_seq_mutation.start_seq, left_flank=obs_seq_mutation.left_flank, right_flank=obs_seq_mutation.right_flank, with_replacement=with_replacement, percent_mutated=float(obs_seq_mutation.num_mutations)/obs_seq_mutation.seq_len)
            raw_start_seq = sample.left_flank + sample.start_seq + sample.right_flank
            raw_end_seq = sample.left_flank + sample.end_seq + sample.right_flank

            start_seq, end_seq, collapse_list = process_degenerates_and_impute_nucleotides(
                raw_start_seq,
                raw_end_seq,
                motif_len,
            )

            sim_seq_mutation = ObservedSequenceMutations(
                start_seq=start_seq,
                end_seq=end_seq,
                motif_len=motif_len,
                collapse_list=collapse_list,
            )
            if sim_seq_mutation.num_mutations > 0:
                # don't consider pairs where mutations occur in flanking regions
                simulated_data.append(sim_seq_mutation)
        return simulated_data

class SurvivalModelSimulatorSingleColumn(SurvivalModelSimulator):
    """
    A simple model that will mutate sequences based on the survival model we've assumed.
    We will suppose that the hazard is constant over time.
    """
    def __init__(self, thetas, probability_matrix, feature_generator, lambda0):
        """
        @param thetas: Numpy array(p, 1) where p is number of motifs. Each element is the rate that the
                    motif mutates.
        @param probability_matrix: The probability of a target nucleotide given that the motif mutated.
        @param feature_generator: FeatureGenerator
        @param lambda0: a constant hazard rate
        """
        self.thetas = thetas
        self.probability_matrix = probability_matrix
        self.feature_generator = feature_generator
        self.motif_list = self.feature_generator.motif_list
        self.lambda0 = lambda0

    def _sample_mutation(self, feature_vec_dict, intermediate_seq, pos_to_mutate):
        # Fill in all the hazard weights -- only fill in for the positions eligible for mutation
        # Note: shmulate removes all stop codons too, which we will not be doing here.
        hazard_weights = []
        mutate_positions = []
        for p in pos_to_mutate:
            motif_idx = feature_vec_dict[p]
            hazard_weights.append(np.exp(self.thetas[motif_idx,0].sum(axis=0)))
            mutate_positions.append(p)

        # sample the time for the next mutation
        # we do inverse transform sampling - sample from unif and then invert
        unif_sample = np.random.rand(1)
        denom = np.sum(hazard_weights)
        mutate_time_delta = - 1/self.lambda0 * np.log(1 - unif_sample) / denom

        # sample mutation position and target nucleotide
        # this is a multinomial
        sampled_idx = sample_multinomial(hazard_weights)
        mutate_pos = mutate_positions[sampled_idx]
        mutate_feat_idx = feature_vec_dict[mutate_pos]
        nucleotide_target_idx = sample_multinomial(
            self.probability_matrix[mutate_feat_idx,:].sum(axis=0)
        )
        return mutate_time_delta, mutate_pos, NUCLEOTIDES[nucleotide_target_idx]

class SurvivalModelSimulatorMultiColumn(SurvivalModelSimulator):
    """
    A simple model that will mutate sequences based on the survival model we've assumed.
    We will suppose that the hazard is constant over time.
    """
    def __init__(self, thetas, feature_generator, lambda0):
        """
        @param thetas: Numpy array(p,4) where p is number of motifs. In each row, theta_row[i] represents the rate
                        to mutate to the i-th nucleotide in the string NUCLEOTIDES. Note that the theta values
                        for target nucleotides that have the SAME center nucleotide as the motif should be
                        completey ignored. (They should technically be negative infinity).
        @param feature_generator: FeatureGenerator
        @param lambda0: a constant hazard rate
        """
        self.thetas = thetas
        assert(thetas.shape[1] == NUM_NUCLEOTIDES)
        self.feature_generator = feature_generator
        self.motif_list = self.feature_generator.motif_list
        self.lambda0 = lambda0

    def _sample_mutation(self, feature_vec_dict, intermediate_seq, pos_to_mutate):
        # Fill in all the hazard weights -- only fill in for the positions eligible for mutation
        # Note: shmulate removes all stop codons too, which we will not be doing here.
        hazard_weights = []
        target_nucleotides = []
        mutate_positions = []
        for p in pos_to_mutate:
            motif_idxs = feature_vec_dict[p]
            nucleotide_to_mutate = intermediate_seq[p]
            for theta_idx, target_nucleotide in enumerate(NUCLEOTIDES):
                if target_nucleotide != nucleotide_to_mutate:
                    exp_theta = np.exp(self.thetas[motif_idxs, theta_idx].sum())
                    hazard_weights.append(exp_theta)
                    target_nucleotides.append(target_nucleotide)
                    mutate_positions.append(p)

        # sample the time for the next mutation
        # we do inverse transform sampling - sample from unif and then invert
        unif_sample = np.random.rand(1)
        denom = np.sum(hazard_weights)
        mutate_time_delta = - 1/self.lambda0 * np.log(1 - unif_sample) / denom

        # sample mutation position and target nucleotide
        # this is a multinomial
        sampled_idx = sample_multinomial(hazard_weights)
        mutate_pos = mutate_positions[sampled_idx]
        nucleotide_target = target_nucleotides[sampled_idx]
        return mutate_time_delta, mutate_pos, nucleotide_target

class SurvivalModelSimulatorPositionDependent(SurvivalModelSimulator):
    """
    A model that will mutate sequences based on a survival model
    incorporating motif and position dependence. We assume the hazard
    is constant over time.
    """
    def __init__(self, thetas, probability_matrix, feature_generator, lambda0, pos_risk):
        """
        @param thetas: Numpy array(p,1) where p is the number of
        motifs. Each element is the rate at which the motif mutates.
        @param probability_matrix: The probability of a target
        nucleotide given that the motif mutated.  @param
        feature_generator: FeatureGenerator @param lambda0: A constant
        hazard rate @param pos_risk: Numpy array(n, 1) where n is the
        length of the sequence. Each element describes how much more
        or less likely the position is to mutate.
        """
        self.thetas = thetas
        self.probability_matrix = probability_matrix
        self.feature_generator = feature_generator
        self.motif_list = self.feature_generator.motif_list
        self.lambda0 = lambda0
        self.pos_risk = pos_risk

    def _sample_mutation(self, feature_vec_dict, intermediate_seq, pos_to_mutate):
        hazard_weights = []
        mutate_positions = []
        for p in pos_to_mutate:
            motif_idx = feature_vec_dict[p]
            hazard_weights.append(np.exp(self.thetas[motif_idx,0].sum(axis=0) + self.pos_risk[p]))
            mutate_positions.append(p)

        unif_sample = np.random.rand(1)
        denom = np.sum(hazard_weights)
        mutate_time_delta = - 1/self.lambda0 * np.log(1 - unif_sample) / denom

        sampled_idx = sample_multinomial(hazard_weights)
        mutate_pos = mutate_positions[sampled_idx]
        mutate_feat_idx = feature_vec_dict[mutate_pos]
        nucleotide_target_idx = sample_multinomial(
            self.probability_matrix[mutate_feat_idx,:].sum(axis=0)
        )
        return mutate_time_delta, mutate_pos, NUCLEOTIDES[nucleotide_target_idx]
