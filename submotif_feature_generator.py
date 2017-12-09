import time
import numpy as np
import itertools
import scipy.sparse
import re
import logging as log

from common import *
from feature_generator import *
from profile_support import profile

class SubmotifFeatureGenerator(FeatureGenerator):
    """
    This makes motifs of the same length (must be odd) and same left flank length
    """
    def __init__(
            self,
            motif_len=3,
            motifs_to_remove=[],
            left_motif_flank_len=1,
            hier_offset=0,
            left_update_region=1,
            right_update_region=1
        ):
        """
        @param motif_len: length of motif
        @param motifs_to_remove: motifs to remove
        @param left_motif_flank_len: length of region to the left of the mutating position
        @param hier_offset: where to offset sequence if we are using hierarchical motifs
        @param left_update_region: number of positions to consider left of mutating position to update
            features correctly
        @param right_update_region: number of positions to consider right of mutating position to update
            features correctly
        """
        assert(left_motif_flank_len in range(motif_len))
        assert(hier_offset >= 0)

        self.motif_len = motif_len
        self.left_motif_flank_len = left_motif_flank_len
        self.right_motif_flank_len = motif_len - left_motif_flank_len - 1

        self.hier_offset = hier_offset

        self.left_update_region = left_update_region
        self.right_update_region = right_update_region

        self.feature_vec_len = np.power(4, motif_len)

        motifs_to_remove = [m for m in motifs_to_remove if len(m) == motif_len]
        self.motif_list = self._get_motif_list(motifs_to_remove)
        self.motif_dict = {motif: i for i, motif in enumerate(self.motif_list)}
        for m in motifs_to_remove:
            self.motif_dict[m] = None

        self.feature_vec_len = len(self.motif_list)

        self.motifs_fused_lasso1 = []
        self.motifs_fused_lasso2 = []

    def create_for_sequence(self, seq_str, left_flank, right_flank, do_feat_vec_pos=None):
        feat_vec_dict = dict()
        seq_len = len(seq_str)
        if do_feat_vec_pos is None:
            do_feat_vec_pos = range(len(seq_str))
        # we'll absorb most of our offsetting into changing the left flank starting position
        left_flank = left_flank[self.hier_offset:]
        # don't generate any feature vector for positions in no_feat_vec_pos since it is not in the risk group
        for pos in do_feat_vec_pos:
            feat_vec_dict[pos] = self._create_feature_vec_for_pos(pos, seq_str, seq_len, left_flank, right_flank)
        return feat_vec_dict

    def _get_base_features(self, obs_seq_mutation):
        """
        Create the feature matrices and feature vector dictionary
        before any mutations have occurred

        @return ObservedSequenceMutations augmented with a feature matrix and dictionary
        """
        indices = []
        start_idx = 0
        indptr = [start_idx]
        num_entries = 0

        for pos in range(obs_seq_mutation.seq_len):
            submotif = obs_seq_mutation.start_seq_with_flanks[pos + self.hier_offset: pos + self.hier_offset + self.motif_len]
            feat_idx = self.motif_dict[submotif]
            if feat_idx is not None:
                start_idx += 1
                indices.append(feat_idx)
            indptr.append(start_idx)

        data = [True] * start_idx
        feat_matrix = scipy.sparse.csr_matrix(
            (data, indices, indptr),
            shape=(obs_seq_mutation.seq_len, self.feature_vec_len),
            dtype=bool,
        )
        return feat_matrix

    def create_base_features(self, obs_seq_mutation):
        """
        Create the feature matrices and feature vector dictionary
        before any mutations have occurred

        @return ObservedSequenceMutations augmented with a feature matrix and dictionary
        """
        assert(obs_seq_mutation.motif_len == self.motif_len)
        indices = []
        start_idx = 0
        indptr = [start_idx]
        num_entries = 0

        for pos in range(obs_seq_mutation.seq_len):
            submotif = obs_seq_mutation.start_seq_with_flanks[pos: pos + self.motif_len]
            feat_idx = self.motif_dict[submotif]
            if feat_idx is not None:
                start_idx += 1
                indices.append(feat_idx)
            indptr.append(start_idx)

        data = [True] * start_idx
        feat_matrix = scipy.sparse.csr_matrix(
            (data, indices, indptr),
            shape=(obs_seq_mutation.seq_len, self.feature_vec_len),
            dtype=bool,
        )

        obs_seq_mutation.set_start_feats(feat_matrix)

        return obs_seq_mutation

    def count_mutated_motifs(self, seq_mut_order):
        """
        Just return back the motifs that mutated for this mutation order

        @param seq_mut_order: ImputedSequenceMutations

        @return list of motifs
        """
        mutated_motifs = []

        old_mutation_pos = None
        intermediate_seq = seq_mut_order.obs_seq_mutation.start_seq

        left_flank = seq_mut_order.obs_seq_mutation.left_flank
        right_flank = seq_mut_order.obs_seq_mutation.right_flank
        left_flank = left_flank[self.hier_offset:]

        seq_len = seq_mut_order.obs_seq_mutation.seq_len
        feat_dict_prev = dict()
        already_mutated_pos = set()
        for pos in seq_mut_order.mutation_order:
            motif_idx = self._create_feature_vec_for_pos(pos, intermediate_seq, seq_len, left_flank, right_flank)
            mutated_motifs.append(self.motif_list[motif_idx])
            # Apply mutation
            intermediate_seq = mutate_string(
                intermediate_seq,
                pos,
                seq_mut_order.obs_seq_mutation.end_seq[pos]
            )

        return mutated_motifs

    def create_for_mutation_steps(self, seq_mut_order):
        """
        Calculate the feature values for the mutation steps
        Only returns the deltas at each mutation step

        @param seq_mut_order: ImputedSequenceMutations

        @return list of FeatureMutationStep (correponding to after first mutation to before last mutation)
        """
        feat_mutation_steps = []

        old_mutation_pos = None
        intermediate_seq = seq_mut_order.obs_seq_mutation.start_seq_with_flanks[self.hier_offset:]

        feat_dict_prev = dict()
        already_mutated_pos = set()
        for mutation_step, mutation_pos in enumerate(seq_mut_order.mutation_order):
            mutating_pos_feat, feat_dict_curr, feat_dict_future = self._update_mutation_step(
                mutation_step,
                mutation_pos,
                old_mutation_pos,
                seq_mut_order,
                intermediate_seq,
                already_mutated_pos,
            )
            feat_mutation_steps.append(MultiFeatureMutationStep(
                mutating_pos_feat,
                mutation_pos,
                neighbors_feat_old=feat_dict_prev,
                neighbors_feat_new=feat_dict_curr,
            ))

            # Apply mutation
            intermediate_seq = mutate_string(
                intermediate_seq,
                mutation_pos + self.left_motif_flank_len,
                seq_mut_order.obs_seq_mutation.end_seq[mutation_pos]
            )
            already_mutated_pos.add(mutation_pos)
            feat_dict_prev = feat_dict_future
            old_mutation_pos = mutation_pos

        if len(feat_mutation_steps) != seq_mut_order.obs_seq_mutation.num_mutations:
            raise AssertionError("%d vs %d" % (len(feat_mutation_steps), seq_mut_order.obs_seq_mutation.num_mutations))
        return feat_mutation_steps

    def create_remaining_mutation_steps(
        self,
        seq_mut_order,
        update_step_start,
    ):
        """
        Calculate the feature values for the mutation steps starting the the `update_step_start`-th step
        Only returns the deltas at each mutation step

        @param seq_mut_order: ImputedSequenceMutations
        @param update_step_start: which mutation step to start calculating features for

        @return list of FeatureMutationStep (correponding to after `update_step_start`-th mutation
                    to before last mutation)
        """
        feat_mutation_steps = []

        old_mutation_pos = None
        feat_dict_prev = dict()
        flanked_seq = seq_mut_order.get_seq_at_step(update_step_start, flanked=True)
        flanked_seq = flanked_seq[self.hier_offset:]

        already_mutated_pos = set(seq_mut_order.mutation_order[:update_step_start])
        for mutation_step in range(update_step_start, seq_mut_order.obs_seq_mutation.num_mutations):
            mutation_pos = seq_mut_order.mutation_order[mutation_step]
            mutating_pos_feat, feat_dict_curr, feat_dict_future = self._update_mutation_step(
                mutation_step,
                mutation_pos,
                old_mutation_pos,
                seq_mut_order,
                flanked_seq,
                already_mutated_pos,
            )
            # print mutation_step, "mutating_pos_feat, feat_dict_curr, feat_dict_future", mutating_pos_feat, feat_dict_curr, feat_dict_future
            feat_mutation_steps.append(MultiFeatureMutationStep(
                mutating_pos_feat,
                mutation_pos,
                neighbors_feat_old=feat_dict_prev,
                neighbors_feat_new=feat_dict_curr,
            ))

            # Apply mutation
            flanked_seq = mutate_string(
                flanked_seq,
                mutation_pos + self.left_motif_flank_len,
                seq_mut_order.obs_seq_mutation.end_seq[mutation_pos]
            )
            already_mutated_pos.add(mutation_pos)
            feat_dict_prev = feat_dict_future
            old_mutation_pos = mutation_pos
        return feat_mutation_steps

    def get_shuffled_mutation_steps_delta(
        self,
        seq_mut_order,
        update_step,
        flanked_seq,
        already_mutated_pos,
    ):
        """
        @param seq_mut_order: a list of the positions in the mutation order
        @param update_step: the index of the mutation step being shuffled with the (`update_step` + 1)-th step
        @param flanked_seq: must be a FLANKED sequence
        @param already_mutated_pos: set of positions that already mutated - dont calculate feature vals for these

        @return a tuple with the feature at this mutation step and the feature mutation step of the next mutation step
        """
        feat_mutation_steps = []
        first_mutation_pos = seq_mut_order.mutation_order[update_step]
        second_mutation_pos = seq_mut_order.mutation_order[update_step + 1]

        first_mut_pos_feat, _, feat_dict_future = self._update_mutation_step(
            update_step,
            first_mutation_pos,
            None,
            seq_mut_order,
            flanked_seq,
            already_mutated_pos,
        )

        # Apply mutation
        flanked_seq = mutate_string(
            flanked_seq,
            first_mutation_pos + self.left_motif_flank_len,
            seq_mut_order.obs_seq_mutation.end_seq[first_mutation_pos]
        )

        second_mut_pos_feat, feat_dict_curr, _ = self._update_mutation_step(
            update_step + 1,
            second_mutation_pos,
            first_mutation_pos,
            seq_mut_order,
            flanked_seq,
            already_mutated_pos,
            calc_future_dict=False,
        )

        return first_mut_pos_feat, MultiFeatureMutationStep(
            second_mut_pos_feat,
            second_mutation_pos,
            neighbors_feat_old=feat_dict_future,
            neighbors_feat_new=feat_dict_curr,
        )

    def _update_mutation_step(
            self,
            mutation_step,
            mutation_pos,
            old_mutation_pos,
            seq_mut_order,
            intermediate_seq,
            already_mutated_pos,
            calc_future_dict=True,
        ):
        """
        Does the heavy lifting for calculating feature vectors at a given mutation step
        @param mutation_step: mutation step index
        @param mutation_pos: the position that is mutating
        @param old_mutation_pos: the position that mutated previously - None if this is first mutation
        @param seq_mut_order: ImputedSequenceMutations
        @param intermediate_seq: nucleotide sequence INCLUDING flanks - before the mutation step occurs

        @return tuple with
            1. the feature index of the position that mutated
            2. a dict with the positions next to the previous mutation and their feature index
            3. a dict with the positions next to the current mutation and their feature index
        """
        mutating_pos_motif = intermediate_seq[mutation_pos: mutation_pos + self.motif_len]
        mutating_pos_feat = self.motif_dict[mutating_pos_motif]

        feat_dict_curr = dict()
        feat_dict_future = dict()
        # Calculate features for positions in the risk group at the time of this mutation step
        # Only requires updating feature values that were close to the previous mutation
        # Get the feature vectors for the positions that might be affected by the previous mutation
        if old_mutation_pos is not None:
            feat_dict_curr = self._get_feature_dict_for_region(
                old_mutation_pos,
                intermediate_seq,
                seq_mut_order.obs_seq_mutation.seq_len,
                already_mutated_pos,
            )

        # Calculate the features in these special positions for updating the next mutation step's risk group
        # Get the feature vectors for the positions that will be affected by current mutation
        if calc_future_dict:
            feat_dict_future = self._get_feature_dict_for_region(
                mutation_pos,
                intermediate_seq,
                seq_mut_order.obs_seq_mutation.seq_len,
                already_mutated_pos,
            )
        return mutating_pos_feat, feat_dict_curr, feat_dict_future

    def _get_feature_dict_for_region(
        self,
        position,
        intermediate_seq,
        seq_len,
        already_mutated_pos,
    ):
        """
        @param position: the position around which to calculate the feature indices for
        @param intermediate_seq: the nucleotide sequence
        @param seq_len: the length of this sequence
        @param already_mutated_pos: which positions already mutated - dont calculate features for these positions

        @return a dict with the positions next to the given position and their feature index
        """
        feat_dict = dict()
        start_region_idx = max(position - self.left_update_region, 0)
        end_region_idx = min(position + self.right_update_region, seq_len - 1)
        update_positions = range(start_region_idx, position) + range(position + 1, end_region_idx + 1)
        for pos in update_positions:
            if pos not in already_mutated_pos:
                # Only update the positions that are in the risk group (the ones that haven't mutated yet)
                submotif = intermediate_seq[pos: pos + self.motif_len]
                feat_idx = self.motif_dict[submotif]
                feat_dict[pos] = feat_idx
        return feat_dict

    def _create_feature_vec_for_pos(self, pos, intermediate_seq, seq_len, left_flank, right_flank):
        """
        @param pos: mutating position
        @param intermediate_seq: intermediate sequence to determine motif, flanks removed
        @param left flank: left flank nucleotide information
        @param right flank: right flank nucleotide information

        Create features for subsequence using information from flanks.
        """
        # if motif length is one then submotifs will be single nucleotides and position remains unchanged
        if pos < self.left_motif_flank_len:
            submotif = left_flank[pos:] + intermediate_seq[:pos + self.right_motif_flank_len + 1]
        elif pos >= seq_len - self.right_motif_flank_len:
            submotif = intermediate_seq[pos - self.left_motif_flank_len:] + right_flank[:pos - seq_len + self.right_motif_flank_len + 1]
        else:
            submotif = intermediate_seq[pos - self.left_motif_flank_len: pos + self.right_motif_flank_len + 1]

        return self.motif_dict[submotif]

    def _get_motif_list(self, motifs_to_remove):
        motif_list = itertools.product(*([NUCLEOTIDES] * self.motif_len))
        filtered_motif_list = []
        for m in motif_list:
            motif = "".join(m)
            if motif not in motifs_to_remove:
                filtered_motif_list.append(motif)

        return filtered_motif_list

    def get_similar_motifs(self, fuse_windows=[3,4], fuse_center_only=False):
        """
        @param fuse_windows: a list of submotif lengths to fuse together -- if two motifs share a submotif of this length,
                            consider these motifs similar

        @return two list of motifs that should be fused together (list1 index i fuses with list2 index i). Motifs
                that differ by one character should be fused together. Also fuses motifs that share central k-mers.
        """
        def _add_grouped_motifs(linked_motifs, grouped_motifs):
            for k, motif_idx_list in grouped_motifs.iteritems():
                for k, m1 in enumerate(motif_idx_list[:-1]):
                    for m2 in motif_idx_list[k+1:]:
                        linked_motifs.add((m1, m2))

        def _get_fuse_motifs(fuse_func):
            fuse_motif_dict = dict()
            for motif_idx, m in enumerate(self.motif_list):
                fuse_motif = fuse_func(m)
                if fuse_motif not in fuse_motif_dict:
                    fuse_motif_dict[fuse_motif] = [motif_idx]
                else:
                    fuse_motif_dict[fuse_motif].append(motif_idx)
            return fuse_motif_dict

        # We implement the fused penalty in terms of differences of pairs that are stored in these
        # index lists: the first entry of the first list minus the first entry in the second list, etc.
        if len(self.motifs_fused_lasso1) == 0:
            log.info("Finding similar motifs")

            linked_motifs = set()

            # Find motifs that share an inner submotif with length from fuse_windows
            for window_len in fuse_windows:
                if not fuse_center_only:
                    for start_idx in range(max(self.motif_len - window_len + 1, 0)):
                        fuse_motif_dict = _get_fuse_motifs(lambda m: m[start_idx:start_idx + window_len])
                        _add_grouped_motifs(linked_motifs, fuse_motif_dict)
                elif window_len % 2 == 1:
                    start_idx = self.left_motif_flank_len - window_len/2
                    fuse_motif_dict = _get_fuse_motifs(lambda m: m[start_idx:start_idx + window_len])
                    _add_grouped_motifs(linked_motifs, fuse_motif_dict)

            # Find motifs that differ by one character
            # Drop the i-th position in the motif and group motifs together by the remaining characters
            # for i in range(self.motif_len):
            #     motifs_diff_one = dict()
            #     for motif_idx, m in enumerate(self.motif_list):
            #         drop_i_motif = m[:i] + m[i+1:]
            #         if drop_i_motif not in motifs_diff_one:
            #             motifs_diff_one[drop_i_motif] = [motif_idx]
            #         else:
            #             motifs_diff_one[drop_i_motif].append(motif_idx)
            #
            #     _add_grouped_motifs(linked_motifs, motifs_diff_one)

            for (m1, m2) in linked_motifs:
                self.motifs_fused_lasso1.append(m1)
                self.motifs_fused_lasso2.append(m2)

        return self.motifs_fused_lasso1, self.motifs_fused_lasso2
