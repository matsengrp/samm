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
            query_motif_list=['agct'],
            cutoff=None,
            max_seq_len=300,
            feats_to_remove=[],
        ):
        """
        @param query_motifs: list of motifs to look for at a distance from mutating position
        @param cutoff: distance from mutating position to look for nearby motif
        """
        assert(query_motif_list)

        if cutoff is None:
            self.cutoff = max_seq_len
        else:
            self.cutoff = cutoff

        self.query_motif_list = query_motif_list
        self.query_motif_list = self._get_motif_list(feats_to_remove)
        self.query_motif_dict = {motif: i for i, motif in enumerate(self.query_motif_list)}

        self.query_motif_len_list = [len(motif) for motif in query_motif_list]
        self.query_motif_lens = set(self.query_motif_len_list)
        self.max_query_motif_len = max(self.query_motif_len_list)

        self.feature_vec_len = len(query_motif_list)

        self.feature_label_list = ['%s within %d' % (motif, self.cutoff) for motif in self.query_motif_list]

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
        start_region_idx = max(position - self.cutoff, 0)
        end_region_idx = min(position + self.cutoff, seq_len)
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
        # get all motifs of certain lengths within cutoff of mutating position
        nearby_motifs = set()
        start_region_idx = max(pos - self.cutoff, 0)
        for query_pos in range(start_region_idx, pos):
            for query_len in self.query_motif_lens:
                nearby_motifs.add(seq_with_flanks[query_pos - query_len + 1: query_pos + 1])
        end_region_idx = min(pos + self.cutoff, len(seq_with_flanks))
        for query_pos in range(pos, end_region_idx):
            for query_len in self.query_motif_lens:
                nearby_motifs.add(seq_with_flanks[query_pos + 1: query_pos + query_len + 1])

        feat_idx = []
        for query_motif in self.query_motif_list:
            if query_motif in nearby_motifs:
                feat_idx.append(self.query_motif_dict[query_motif])

        return feat_idx

    def _get_mutated_seq(self, intermediate_seq, pos, end_seq):
        """
        @param intermediate_seq: initial nucleotide sequence
        @param pos: mutating position
        @param end_seq: final nucleotide sequence

        @return a dict with the positions next to the given position and their feature index
        """
        return mutate_string(
            intermediate_seq,
            pos,
            end_seq[pos],
        )

    def _get_motif_list(self, feats_to_remove):
        filtered_motif_list = []
        motifs_to_remove = [label.split(' ')[0] for label in feats_to_remove]
        for motif in self.query_motif_list:
            if motif not in motifs_to_remove:
                filtered_motif_list.append(motif)
        return filtered_motif_list

    def _update_motifs_after_removing(self, feats_to_remove=[]):
        """
        take existing SubmotifGenerator and update it with features to remove
        """
        self.query_motif_list = self._get_motif_list(feats_to_remove)
        self.query_motif_dict = {motif: i for i, motif in enumerate(self.query_motif_list)}

        self.query_motif_len_list = [len(motif) for motif in self.query_motif_list]
        self.query_motif_lens = set(self.query_motif_len_list)
        self.max_query_motif_len = max(self.query_motif_len_list)

        self.feature_vec_len = len(self.query_motif_list)

        self.feature_label_list = ['%s within %d' % (motif, self.cutoff) for motif in self.query_motif_list]
