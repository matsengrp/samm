import numpy as np
import itertools
import logging as log

from common import mutate_string
from generic_feature_generator import GenericFeatureGenerator

class SubmotifFeatureGenerator(GenericFeatureGenerator):
    """
    This makes motifs of the same length (must be odd) and same left flank length
    """
    def __init__(
            self,
            motif_len=3,
            feats_to_remove=[],
            left_motif_flank_len=1,
            hier_offset=0,
            left_update_region=1,
            right_update_region=1
        ):
        """
        @param motif_len: length of motif
        @param feats_to_remove: motifs to remove
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

        # add properties of feature generator
        self._update_motifs_after_removing(feats_to_remove)

        self.motifs_fused_lasso1 = []
        self.motifs_fused_lasso2 = []

    def _update_motifs_after_removing(self, feats_to_remove=[]):
        """
        take existing SubmotifGenerator and update it with features to remove
        """
        motifs_to_remove = [m[:self.motif_len] for m in feats_to_remove]
        self.motif_list = self._get_motif_list(motifs_to_remove)
        self.motif_dict = {motif: i for i, motif in enumerate(self.motif_list)}
        for m in motifs_to_remove:
            self.motif_dict[m] = None

        self.feature_label_list = ['%s, position %d' % (motif, self.left_motif_flank_len) for motif in self.motif_list]

        self.feature_vec_len = len(self.motif_list)

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
                feat_dict[pos] = _get_mutating_pos_feat_idx(pos, intermediate_seq)
        return feat_dict

    def _get_mutating_pos_feat_idx(self, pos, seq_with_flanks):
        """
        @param pos: mutating position
        @param seq_with_flanks: sequence to determine motif, flanks included

        @return index of feature vector for this mutating position
        """
        submotif = seq_with_flanks[pos + self.hier_offset: pos + self.hier_offset + self.motif_len]
        return [self.motif_dict[submotif]]

    def _get_mutated_seq(self, intermediate_seq, pos, end_seq):
        """
        @param intermediate_seq: initial nucleotide sequence
        @param pos: mutating position
        @param end_seq: final nucleotide sequence

        @return a dict with the positions next to the given position and their feature index
        """
        return mutate_string(
            intermediate_seq[self.hier_offset:],
            pos + self.left_motif_flank_len,
            end_seq[pos],
        )

    def _get_motif_list(self, motifs_to_remove):
        motif_list = itertools.product(*([NUCLEOTIDES] * self.motif_len))
        filtered_motif_list = []
        for m in motif_list:
            motif = "".join(m)
            if motif not in motifs_to_remove:
                filtered_motif_list.append(motif)

        return filtered_motif_list

    def count_mutated_motifs(self, seq_mut_order):
        """
        Just return back the motifs that mutated for this mutation order

        @param seq_mut_order: ImputedSequenceMutations

        @return list of motifs
        """
        mutated_motifs = []

        old_mutation_pos = None
        intermediate_seq = seq_mut_order.obs_seq_mutation.start_seq_with_flanks

        seq_len = seq_mut_order.obs_seq_mutation.seq_len
        feat_dict_prev = dict()
        already_mutated_pos = set()
        for pos in seq_mut_order.mutation_order:
            mutated_motifs.append(
                intermediate_seq[pos + self.hier_offset: pos + self.hier_offset + self.motif_len]
            )
            # Apply mutation
            intermediate_seq = self._get_mutated_seq(
                intermediate_seq,
                pos,
                seq_mut_order.obs_seq_mutation.end_seq,
            )

        return mutated_motifs

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
