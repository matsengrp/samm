import time
import numpy as np
from common import *
import itertools
from profile_support import profile
import scipy.sparse

from feature_generator import FeatureGenerator
from models import SequencePositionFeatures

class MotifFeatureGenerator(FeatureGenerator):
    def __init__(self, motif_len=3):
        assert(motif_len % 2 == 1)
        self.motif_len = motif_len
        self.flank_end_len = motif_len/2
        self.feature_vec_len = np.power(4, motif_len) + 1

        motif_list = self.get_motif_list()
        self.motif_dict = {motif: i for i, motif in enumerate(motif_list)}

    @profile
    def update_for_sequence(self, seq_pos_feats, no_feat_vec_pos, do_feat_vec_pos, theta=None):
        """
        @param seq_pos_feats: SequencePositionFeatures
        @returns SequencePositionFeatures
        """
        update_pos_idxs = do_feat_vec_pos - no_feat_vec_pos
        for pos in update_pos_idxs:
            feat_vec_idxs = self._get_feature_idxs(pos, seq_pos_feats.seq_str)
            seq_pos_feats.update_feat_matrix(pos, feat_vec_idxs)

        if theta is not None:
            seq_pos_feats.update_theta_sums(theta, update_pos_idxs)

    @profile
    def create_for_sequence(self, obs_seq_mutation, theta=None, is_sparse=True):
        """
        @param obs_seq_mutation: ObservedSequenceMutations
        @returns SequencePositionFeatures
        """
        # feat_matrix = scipy.sparse.lil_matrix((obs_seq_mutation.seq_len, self.feature_vec_len), dtype=bool)
        # for pos in range(obs_seq_mutation.seq_len):
        #     feat_vec_idxs = self._get_feature_idxs(pos, obs_seq_mutation.start_seq)
        #     feat_matrix.rows[pos] = feat_vec_idxs
        #     feat_matrix.data[pos] = [True] * feat_vec_idxs.size

        num_entries = 0
        indices = []
        indptr = [0]
        # feat_matrix = scipy.sparse.lil_matrix((obs_seq_mutation.seq_len, self.feature_vec_len), dtype=bool)
        rows = []
        data = []
        for pos in range(obs_seq_mutation.seq_len):
            feat_vec_idxs = self._get_feature_idxs(pos, obs_seq_mutation.start_seq)
            rows.append(feat_vec_idxs)
            num_entries += len(feat_vec_idxs)
            indptr.append(num_entries)

            # rows.append(feat_vec_idxs)
            # data.append([True] * len(feat_vec_idxs))

        data = [True] * num_entries
        indices = np.concatenate(rows)
        csr_matrix = scipy.sparse.csr_matrix((data, indices, indptr), shape=(obs_seq_mutation.seq_len, self.feature_vec_len),dtype=bool)
        # feat_matrix = csr_matrix.tolil()
        feat_matrix = None

        seq_pos_feats = SequencePositionFeatures(
            obs_seq_mutation.start_seq,
            feat_matrix,
            is_sparse,
            np.ones(obs_seq_mutation.seq_len, dtype=bool),
            csr_matrix=csr_matrix,
            # feat_rows=rows,
            # feat_data=data,
        )
        if theta is not None:
            seq_pos_feats.update_theta_sums(theta)

        return seq_pos_feats

    @profile
    def create_for_mutation_steps(self, seq_mut_order, theta=None):
        num_steps = seq_mut_order.obs_seq_mutation.num_mutations - 1

        seq_pos_feats = self.create_for_sequence(seq_mut_order.obs_seq_mutation, theta=theta)
        mutation_risk_groups = [seq_pos_feats]
        no_feat_vec_pos = set()
        for i, mutation_pos in enumerate(seq_mut_order.mutation_order[:-1]):
            # print "mutation_pos", mutation_pos
            # Make a copy of the features from the previous mutation step and update it
            seq_pos_feats_new = seq_pos_feats.copy()


            # mutation_pos is the position that just mutated (so no longer in the risk group)
            seq_pos_feats_new.inactivate_position(mutation_pos)

            # update to get the new sequence after the i-th mutation
            seq_pos_feats_new.seq_str = mutate_string(
                seq_pos_feats_new.seq_str,
                mutation_pos,
                seq_mut_order.obs_seq_mutation.mutation_pos_dict[mutation_pos]
            )

            # Get the feature vectors for the positions that might be affected by the latest mutation
            # Don't calculate feature vectors for positions that have mutated already
            no_feat_vec_pos = no_feat_vec_pos.union([mutation_pos])
            # print "no_feat_vec_pos", no_feat_vec_pos
            # Calculate feature vectors for positions that are close to the previous mutation
            do_feat_vec_pos = set(range(
                max(mutation_pos - self.flank_end_len, 0),
                min(mutation_pos + self.flank_end_len + 1, seq_mut_order.obs_seq_mutation.seq_len),
            ))
            self.update_for_sequence(
                seq_pos_feats_new,
                no_feat_vec_pos=no_feat_vec_pos,
                do_feat_vec_pos=do_feat_vec_pos,
                theta=theta
            )

            mutation_risk_groups.append(seq_pos_feats_new)
            seq_pos_feats = seq_pos_feats_new

        return mutation_risk_groups

    def update_for_mutation_steps(
        self,
        seq_mut_order,
        update_steps,
        base_mutation_risk_groups,
        theta=None
    ):
        mutation_risk_groups = list(base_mutation_risk_groups)
        no_feat_vec_pos = set(seq_mut_order.mutation_order[:update_steps[0]])
        for i in update_steps:
            mutation_pos = seq_mut_order.mutation_order[i]
            # Make a copy of the features from the previous mutation step and update it
            mutation_risk_groups[i+1] = mutation_risk_groups[i].copy()


            # mutation_pos is the position that just mutated (so no longer in the risk group)
            mutation_risk_groups[i+1].inactivate_position(mutation_pos)

            # update to get the new sequence after the i-th mutation
            mutation_risk_groups[i+1].seq_str = mutate_string(
                mutation_risk_groups[i+1].seq_str,
                mutation_pos,
                seq_mut_order.obs_seq_mutation.mutation_pos_dict[mutation_pos]
            )

            # Get the feature vectors for the positions that might be affected by the latest mutation
            # Don't calculate feature vectors for positions that have mutated already
            no_feat_vec_pos = no_feat_vec_pos.union([mutation_pos])
            # print "no_feat_vec_pos", no_feat_vec_pos
            # Calculate feature vectors for positions that are close to the previous mutation
            do_feat_vec_pos = set(range(
                max(mutation_pos - self.flank_end_len, 0),
                min(mutation_pos + self.flank_end_len + 1, seq_mut_order.obs_seq_mutation.seq_len),
            ))
            self.update_for_sequence(
                mutation_risk_groups[i+1],
                no_feat_vec_pos=no_feat_vec_pos,
                do_feat_vec_pos=do_feat_vec_pos,
                theta=theta
            )

        return mutation_risk_groups

    def _get_feature_idxs(self, pos, intermediate_seq):
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
