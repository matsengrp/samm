import time
import numpy as np
from common import *
import itertools
import scipy.sparse

from feature_generator import *
from profile_support import profile

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

    def create_base_features(self, obs_seq_mutation):
        """
        Create the feature matrices and feature vector dictionary
        before any mutations have occurred

        @return ObservedSequenceMutationsFeatures
        """
        indices = np.array([])
        indptr = [0]
        num_entries = 0

        feat_dict = dict()
        for pos in range(obs_seq_mutation.seq_len):
            feat_vect = self._create_feature_vec_for_pos(pos, obs_seq_mutation.start_seq)
            feat_dict[pos] = feat_vect
            num_entries += feat_vect.size
            indptr.append(num_entries)
            indices = np.concatenate((indices, feat_vect))

        data = [True] * num_entries
        feat_matrix = scipy.sparse.csr_matrix(
            (data, indices, indptr),
            shape=(obs_seq_mutation.seq_len, self.feature_vec_len),
            dtype=bool,
        )
        return ObservedSequenceMutationsFeatures(
            obs_seq_mutation,
            feat_matrix,
            feat_dict,
        )

    def create_for_sequence(self, sequence, no_feat_vec_pos=set(), do_feat_vec_pos=None):
        feature_vec_dict = dict()
        if do_feat_vec_pos is None:
            do_feat_vec_pos = set(range(len(sequence)))

        # don't generate any feature vector for positions in no_feat_vec_pos since it is not in the risk group
        for pos in do_feat_vec_pos-no_feat_vec_pos:
            feature_vec_dict[pos] = self._create_feature_vec_for_pos(pos, sequence)

        return feature_vec_dict

    # TODO: make this a real feature generator (deal with ends properly)
    def create_for_mutation_steps(self, seq_mut_order, theta=None):
        num_steps = seq_mut_order.obs_seq_mutation.num_mutations - 1
        feat_mutation_steps = FeatureMutationSteps(seq_mut_order)

        feature_vec_thetasums = None
        if theta is not None:
            assert(theta.shape[1] == 1)
            theta_sum0 = feat_mutation_steps.get_init_theta_sum(theta)
            feature_vec_thetasums = [
                { p : theta_sum0[p, 0] for p in range(seq_mut_order.obs_seq_mutation.seq_len)}
            ] + [None] * num_steps

        no_feat_vec_pos = set()
        for i, mutation_pos in enumerate(seq_mut_order.mutation_order[:-1]):
            no_feat_vec_pos.add(mutation_pos)
            seq_str, feat_dict, thetasum = self._update_mutation_step(
                i,
                mutation_pos,
                feat_mutation_steps,
                feature_vec_thetasums,
                seq_mut_order,
                no_feat_vec_pos,
                theta,
            )
            feat_mutation_steps.update(i + 1, seq_str, feat_dict)
            if theta is not None:
                feature_vec_thetasums[i + 1] = thetasum

        return feat_mutation_steps, feature_vec_thetasums

    # TODO: make this a real feature generator (deal with ends properly)

    def update_for_mutation_steps(
        self,
        seq_mut_order,
        update_steps,
        base_feat_mutation_steps,
        base_feature_vec_theta_sums=None,
        theta=None
    ):
        num_steps = seq_mut_order.obs_seq_mutation.num_mutations
        feat_mutation_steps = base_feat_mutation_steps.copy()

        feature_vec_thetasums = None
        if theta is not None:
            feature_vec_thetasums = list(base_feature_vec_theta_sums)

        no_feat_last_idx = 0
        no_feat_vec_pos = set()
        for i in update_steps:
            if i >= num_steps - 1:
                break

            mutation_pos = seq_mut_order.mutation_order[i]

            no_feat_vec_pos.update(seq_mut_order.mutation_order[no_feat_last_idx:i + 1])
            no_feat_last_idx = i + 1

            seq_str, feat_dict, thetasum = self._update_mutation_step(
                i,
                mutation_pos,
                feat_mutation_steps,
                feature_vec_thetasums,
                seq_mut_order,
                no_feat_vec_pos,
                theta,
            )
            feat_mutation_steps.update(i + 1, seq_str, feat_dict)
            if theta is not None:
                feature_vec_thetasums[i + 1] = thetasum

        return feat_mutation_steps, feature_vec_thetasums

    def _update_mutation_step(self, i, mutation_pos, feat_mutation_steps, feature_vec_thetasums, seq_mut_order, no_feat_vec_pos, theta=None):
        """
        Does the heavy lifting for calculating feature vectors at a given mutation step
        @param i: mutation step index
        @param mutation_pos: the position that is mutating
        @param feat_mutation_steps: FeatureMutationSteps
        @param feature_vec_thetasums: theta sums (should be None if theta is None)
        @param seq_mut_order: the observed mutation order information (ObservedSequenceMutationsFeatures)
        @param no_feat_vec_pos: the positions that mutated already and should not have calculations performed for them
        @param theta: if passed in, calculate theta sum values too

        @return new_intermediate_seq, feat_vec_dict_next, feature_vec_thetasum_next
        """
        # update to get the new sequence after the i-th mutation
        new_intermediate_seq = mutate_string(
            feat_mutation_steps.intermediate_seqs[i],
            mutation_pos,
            seq_mut_order.obs_seq_mutation.mutation_pos_dict[mutation_pos]
        )
        # Get the feature vectors for the positions that might be affected by the latest mutation
        # Don't calculate feature vectors for positions that have mutated already
        # Calculate feature vectors for positions that are close to the mutation
        do_feat_vec_pos = set(range(
            max(mutation_pos - self.flank_end_len, 0),
            min(mutation_pos + self.flank_end_len + 1, seq_mut_order.obs_seq_mutation.seq_len),
        ))
        feat_vec_dict_update = self.create_for_sequence(
            new_intermediate_seq,
            no_feat_vec_pos=no_feat_vec_pos,
            do_feat_vec_pos=do_feat_vec_pos,
        )

        feat_vec_dict_next = feat_mutation_steps.feature_vec_dicts[i].copy() # shallow copy of dictionary
        feat_vec_dict_next.pop(mutation_pos, None)
        feat_vec_dict_next.update(feat_vec_dict_update)

        feature_vec_thetasum_next = None
        if theta is not None:
            feature_vec_thetasum_next = feature_vec_thetasums[i].copy() # shallow copy of dictionary
            feature_vec_thetasum_next.pop(mutation_pos, None)
            for p, feat_vec in feat_vec_dict_update.iteritems():
                ### TODO: IS THIS CORRECT FOR MULTIPLE THETA COLUMNS
                feature_vec_thetasum_next[p] = get_theta_sum_mask(theta, feat_vec)

        return new_intermediate_seq, feat_vec_dict_next, feature_vec_thetasum_next

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
        return np.array([idx], dtype=np.intp)

    def get_motif_list(self):
        motif_list = itertools.product(*([NUCLEOTIDES] * self.motif_len))
        return ["".join(m) for m in motif_list]
