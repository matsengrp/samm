import time
import numpy as np
from common import *
import itertools

class FeatureGenerator:
    """
    Subclass this to have various types of feature vector generators.
    We assume all feature vectors are composed of ones and zeros.
    Each feature vector will a dictionary keyed on motifs with values being the
    corresponding index in the list of motifs (changing the right-hand bases
    first). We call these "sparse feature vectors".
    """

    def create_for_sequence(self, sequence, no_feat_vec_pos=[], do_feat_vec_pos=None):
        """
        @param sequence: current sequence (string)
        @param no_feat_vec_pos: do NOT generate feature vectors for these positions
        @param do_feat_vec_pos: do generate feature vectors for these positions.
                                By default, this will be set to all positions in the sequence.
        @return: list of sparse feature vectors

        If there are positions in both no_feat_vec_pos and do_feat_vec_pos, then
        no feature vector is generated for that position.
        """
        raise NotImplementedError()

    def create_for_mutation_steps(self, seq_mut_order):
        """
        @param seq_mut_order: a ImputedSequenceMutations
        @return: a tuple with:
            1. list of sparse feature vectors for all positions in the risk group after the i-th mutation,
                for all mutation steps in seq_mut_order
            2. a list of all the intermediate sequences that occured throughout the mutation steps
        """
        raise NotImplementedError()

    def update_for_mutation_steps(
        self,
        seq_mut_order,
        update_steps,
        base_feat_vec_dicts,
        base_intermediate_seqs,
    ):
        """
        Calculates a list of feature vectors and intermediate sequences by borrowoing information from
        previously calculated feature vectors and intermediate sequences. By indicating after which mutation
        steps the intermediate sequences differ, we can know which steps need to be calculated vs. which steps
        can just be copied.

        @param seq_mut_order: a ImputedSequenceMutations
        @param update_steps: the indices of the mutation steps where the intermediate sequence is different
        @param base_feat_vec_dicts: a list of sparse feature vectors for all mutation steps
        @param base_intermediate_seqs: a list of all the intermediate sequences for all mutation steps
        @return: a tuple with:
            1. list of sparse feature vectors for all positions in the risk group after the i-th mutation,
                for all mutation steps in seq_mut_order
            2. a list of all the intermediate sequences that occured throughout the mutation steps
        """
        raise NotImplementedError()

class SubmotifFeatureGenerator(FeatureGenerator):
    """
    This makes motifs of the same length (must be odd). For positions on the edge, we currently
    just use an indicator to denote the position is on the edge.
    TODO: Do something else for edge positions?
    """
    def __init__(self, motif_len=3):
        assert(motif_len % 2 == 1)
        self.motif_len = motif_len
        self.flank_end_len = motif_len/2
        self.feature_vec_len = np.power(4, motif_len) + 1

        motif_list = self.get_motif_list()
        self.motif_dict = {motif: i for i, motif in enumerate(motif_list)}

    def create_for_sequence(self, sequence, no_feat_vec_pos=set(), do_feat_vec_pos=None):
        feature_vec_dict = dict()
        return self.update_for_sequence(sequence, no_feat_vec_pos, do_feat_vec_pos)

    def update_for_sequence(self, sequence, no_feat_vec_pos=set(), do_feat_vec_pos=None):
        feature_vec_dict = dict()
        if do_feat_vec_pos is None:
            do_feat_vec_pos = set(range(len(sequence)))

        # don't generate any feature vector for positions in no_feat_vec_pos since it is not in the risk group
        for pos in do_feat_vec_pos-no_feat_vec_pos:
            feature_vec_dict[pos] = self._create_feature_vec_for_pos(pos, sequence)

        return feature_vec_dict

    # TODO: make this a real feature generator (deal with ends properly)
    def create_for_mutation_steps(self, seq_mut_order, theta=None):
        """
        @param seq_mut_order: ImputedSequenceMutations
        @param theta: if a theta vector is given, it will also calculate the theta sum (theta * feature vector) for each position

        @return: 1. list of sparse feature vectors for positions in the risk group after the i-th mutation,
                2. list of intermediate sequences formed at each mutation step
                3. If theta is given, list of theta sums for positions in the risk group at each mutation step. Otherwise None
        """
        num_steps = seq_mut_order.obs_seq_mutation.num_mutations - 1
        intermediate_seq = seq_mut_order.obs_seq_mutation.start_seq
        intermediate_seqs = [intermediate_seq] + [None] * num_steps
        feature_vec_dicts = [self.create_for_sequence(intermediate_seq)] + [None] * num_steps

        feature_vec_thetasums = None
        if theta is not None:
            feature_vec_thetasums = [
                { p : get_theta_sum_mask(theta, feat_vec) for p,feat_vec in feature_vec_dicts[0].iteritems() }
            ] + [None] * num_steps

        for i, mutation_pos in enumerate(seq_mut_order.mutation_order[:-1]):
            # update to get the new sequence after the i-th mutation
            intermediate_seq = mutate_string(
                intermediate_seq,
                mutation_pos,
                seq_mut_order.obs_seq_mutation.mutation_pos_dict[mutation_pos]
            )
            # Get the feature vectors for the positions that might be affected by the latest mutation
            # Don't calculate feature vectors for positions that have mutated already
            no_feat_vec_pos = set(seq_mut_order.mutation_order[:i + 1])
            # Calculate feature vectors for positions that are close to the mutation
            do_feat_vec_pos = set(range(
                max(mutation_pos - self.flank_end_len, 0),
                min(mutation_pos + self.flank_end_len + 1, seq_mut_order.obs_seq_mutation.seq_len),
            ))
            feat_vec_dict_next = feature_vec_dicts[i].copy()
            feat_vec_dict_next.pop(seq_mut_order.mutation_order[i], None)
            feat_vec_dict_update = self.update_for_sequence(
                intermediate_seq,
                no_feat_vec_pos=no_feat_vec_pos,
                do_feat_vec_pos=do_feat_vec_pos,
            )
            feat_vec_dict_next.update(feat_vec_dict_update)
            feature_vec_dicts[i + 1] = feat_vec_dict_next
            intermediate_seqs[i + 1] = intermediate_seq

            if theta is not None:
                feature_vec_thetasum_next = feature_vec_thetasums[i].copy()
                feature_vec_thetasum_next.pop(seq_mut_order.mutation_order[i], None)
                for p, feat_vec in feat_vec_dict_update.iteritems():
                    feature_vec_thetasum_next[p] = get_theta_sum_mask(theta, feat_vec)
                feature_vec_thetasums[i + 1] = feature_vec_thetasum_next

        return feature_vec_dicts, intermediate_seqs, feature_vec_thetasums

    # TODO: make this a real feature generator (deal with ends properly)
    def update_for_mutation_steps(self,
        seq_mut_order,
        update_steps,
        base_feat_vec_dicts,
        base_intermediate_seqs,
        base_feature_vec_theta_sums=None,
        theta=None
    ):
        """
        Returns a feature vec given a list of steps that need to be updated
        Note: This makes a copy of the lists given so it won't modify `base_feat_vec_dicts, base_intermediate_seqs` in place.

        @param seq_mut_order: ImputedSequenceMutations
        @param update_steps: which steps in the mutation sequence are different and need to be updated
        @param base_feat_vec_dicts: list of feature vectors for positions from a similar mutation sequence
        @param base_intermediate_seqs: list of sequences from a similar mutation sequence
        @param base_feature_vec_theta_sums: list of theta sums from a similar mutation sequence
        @param theta: if a theta vector is given, function will also calculate the theta sum (theta * feature vector) for each position

        @return: 1. list of sparse feature vectors for positions in the risk group after the i-th mutation,
                2. list of intermediate sequences formed at each mutation step
                3. If theta and base_feature_vec_theta_sums given, list of theta sums for positions in the risk group at each mutation step. Otherwise None
        """
        num_steps = len(base_intermediate_seqs)
        intermediate_seqs = list(base_intermediate_seqs)
        feature_vec_dicts = list(base_feat_vec_dicts)
        feature_vec_thetasums = None
        if theta is not None and base_feature_vec_theta_sums is not None:
            feature_vec_thetasums = list(base_feature_vec_theta_sums)
        for i in update_steps:
            if i >= num_steps - 1:
                break
            mutation_pos = seq_mut_order.mutation_order[i]
            # update to get the new sequence after the i-th mutation
            intermediate_seqs[i+1] = mutate_string(
                intermediate_seqs[i],
                mutation_pos,
                seq_mut_order.obs_seq_mutation.mutation_pos_dict[mutation_pos]
            )
            # Get the feature vectors for the positions that might be affected by the latest mutation
            # Don't calculate feature vectors for positions that have mutated already
            no_feat_vec_pos = set(seq_mut_order.mutation_order[:i + 1])
            # Calculate feature vectors for positions that are close to the mutation
            do_feat_vec_pos = set(range(
                max(mutation_pos - self.flank_end_len, 0),
                min(mutation_pos + self.flank_end_len + 1, seq_mut_order.obs_seq_mutation.seq_len),
            ))
            feat_vec_dict_next = feature_vec_dicts[i].copy()
            feat_vec_dict_next.pop(seq_mut_order.mutation_order[i], None)
            feat_vec_dict_update = self.update_for_sequence(
                intermediate_seqs[i + 1],
                no_feat_vec_pos=no_feat_vec_pos,
                do_feat_vec_pos=do_feat_vec_pos,
            )
            feat_vec_dict_next.update(feat_vec_dict_update)

            feature_vec_dicts[i + 1] = feat_vec_dict_next

            if theta is not None:
                feature_vec_thetasum_next = feature_vec_thetasums[i].copy()
                feature_vec_thetasum_next.pop(seq_mut_order.mutation_order[i], None)
                for p, feat_vec in feat_vec_dict_update.iteritems():
                    feature_vec_thetasum_next[p] = get_theta_sum_mask(theta, feat_vec)
                feature_vec_thetasums[i + 1] = feature_vec_thetasum_next

        return feature_vec_dicts, intermediate_seqs, feature_vec_thetasums

    def _create_feature_vec_for_pos(self, pos, intermediate_seq):
        """
        @param no_feat_vec_pos: don't create feature vectors for these positions
        """
        submotif = intermediate_seq[pos - self.flank_end_len: pos + self.flank_end_len + 1]
        any_degenerate = any([nucleotide not in NUCLEOTIDES for nucleotide in submotif])
        if pos < self.flank_end_len or pos > len(intermediate_seq) - 1 - self.flank_end_len or any_degenerate:
            # do special stuff cause positions are at the ends or have degenerate bases (N or . usually)
            # TODO: update this. right now it sets all extreme positions to the same feature
            idx = self.feature_vec_len - 1
        else:
            idx = self.motif_dict[submotif]
        return [idx]

    def get_motif_list(self):
        motif_list = itertools.product(*([NUCLEOTIDES] * self.motif_len))
        return ["".join(m) for m in motif_list]
