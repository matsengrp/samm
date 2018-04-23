import numpy as np
import itertools
import logging as log

from common import mutate_string, NUCLEOTIDES
from generic_feature_generator import GenericFeatureGenerator

class AdjacentMotifFeatureGenerator(GenericFeatureGenerator):
    """
    Distances calculated from rightmost nucleotide of query motifs
    """
    def __init__(
            self,
            motif_len=4,
            distance_to_right_flank_end=0,
            feats_to_remove=[],
            left_update_region=1,
            right_update_region=1
        ):
        """
        @param motif_len: list of motifs to look for at a distance from mutating position
        @param distance_to_right_flank_end: distance from mutating position to look for nearby motif
        @param feats_to_remove: motifs to remove
        @param left_update_region: number of positions to consider left of mutating position to update
            features correctly
        @param right_update_region: number of positions to consider right of mutating position to update
            features correctly
        """
        self.motif_len = motif_len
        self.distance_to_right_flank_end = distance_to_right_flank_end
        self.left_update_region = left_update_region
        self.right_update_region = right_update_region

        self.update_feats_after_removing(feats_to_remove)

    def update_feats_after_removing(self, feats_to_remove=[]):
        """
        take existing SubmotifGenerator and update it with features to remove
        """
        all_feature_info_list = [("".join(motif), self.distance_to_right_flank_end) for motif in itertools.product(*([NUCLEOTIDES] * self.motif_len))]
        self.feature_info_list = [feat_tuple for feat_tuple in all_feature_info_list if feat_tuple not in feats_to_remove]

        self.motif_list = [motif for motif, _ in self.feature_info_list]
        self.motif_dict = {motif: i for i, motif in enumerate(self.motif_list)}
        for motif, dist in all_feature_info_list:
            if (motif, dist) in feats_to_remove:
                self.motif_dict[motif] = None

        self.feature_vec_len = len(self.feature_info_list)

    def print_label_from_info(self, info):
        return "adjacent motif: %s, distance: %d" % (info[0], info[1])

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
        update_positions = range(start_region_idx, position) + range(position + 1, end_region_idx)
        for pos in update_positions:
            if pos not in already_mutated_pos:
                # Only update the positions that are in the risk group (the ones that haven't mutated yet)
                feat_dict[pos] = self._get_mutating_pos_feat_idx(pos, intermediate_seq)
        return feat_dict

    def _get_mutating_pos_feat_idx(self, pos, seq_with_flanks):
        """
        @param pos: mutating position
        @param seq_with_flanks: sequence to determine motif, flanks included

        @return index of feature vector for this mutating position
        """
        if pos + self.distance_to_right_flank_end - self.motif_len < 0:
            feat_idx = None
        elif pos + self.distance_to_right_flank_end > len(seq_with_flanks):
            feat_idx = None
        else:
            adj_motif = seq_with_flanks[pos + self.distance_to_right_flank_end - self.motif_len: pos + self.distance_to_right_flank_end]
            feat_idx = self.motif_dict[adj_motif]
        return feat_idx
