import time
import numpy as np
import itertools
import scipy.sparse
import re
import random

from common import *
from feature_generator import *
from profile_support import profile

class SubmotifFastFeatureGenerator(FeatureGenerator):
    """
    This makes motifs of the same length (must be odd).
    """
    def __init__(self, motif_len=1):
        assert(motif_len % 2 == 1)
        self.motif_len = motif_len
        self.half_motif_len = motif_len/2
        self.feature_vec_len = np.power(4, motif_len)

        motif_list = self.get_motif_list()
        self.motif_dict = {motif: i for i, motif in enumerate(motif_list)}

    def create_for_sequence(self, seq_str, left_flank, right_flank, do_feat_vec_pos=None):
        feat_vec_dict = dict()
        seq_len = len(seq_str)
        if do_feat_vec_pos is None:
            do_feat_vec_pos = range(len(seq_str))

        # don't generate any feature vector for positions in no_feat_vec_pos since it is not in the risk group
        for pos in do_feat_vec_pos:
            feat_vec_dict[pos] = self._create_feature_vec_for_pos(pos, seq_str, seq_len, left_flank, right_flank)
        return feat_vec_dict

    def create_base_features(self, obs_seq_mutation):
        """
        Create the feature matrices and feature vector dictionary
        before any mutations have occurred

        @return ObservedSequenceMutations augmented with a feature matrix and dictionary
        """
        indices = []
        indptr = [0]
        num_entries = 0

        feat_dict = dict()
        for pos in range(obs_seq_mutation.seq_len):
            submotif = obs_seq_mutation.start_seq_with_flanks[pos: pos + self.motif_len]
            feat_idx = self.motif_dict[submotif]
            feat_dict[pos] = feat_idx
            indptr.append(pos + 1)
            indices.append(feat_idx)

        data = [True] * obs_seq_mutation.seq_len
        feat_matrix = scipy.sparse.csr_matrix(
            (data, indices, indptr),
            shape=(obs_seq_mutation.seq_len, self.feature_vec_len),
            dtype=bool,
        )

        obs_seq_mutation.feat_matrix_start = feat_matrix
        obs_seq_mutation.feat_dict_start = feat_dict

        return obs_seq_mutation

    @profile
    def create_for_mutation_steps(self, seq_mut_order):
        """
        @param seq_mut_order: ImputedSequenceMutations

        @return list of FeatureMutationStep (correponding to after first mutation to before last mutation)
        """
        feat_mutation_steps = []

        old_mutation_pos = None
        intermediate_seq = seq_mut_order.obs_seq_mutation.start_seq_with_flanks
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
            feat_mutation_steps.append(FeatureMutationStep(
                mutating_pos_feat,
                feat_dict_prev,
                feat_dict_curr,
            ))

            # Apply mutation
            intermediate_seq = mutate_string(
                intermediate_seq,
                mutation_pos + self.half_motif_len,
                seq_mut_order.obs_seq_mutation.end_seq[mutation_pos]
            )
            already_mutated_pos.add(mutation_pos)
            feat_dict_prev = feat_dict_future
            old_mutation_pos = mutation_pos

        assert(len(feat_mutation_steps) == seq_mut_order.obs_seq_mutation.num_mutations)
        return feat_mutation_steps

    def create_remaining_mutation_steps(
        self,
        seq_mut_order,
        update_step_start,
    ):
        """
        @param seq_mut_order: ImputedSequenceMutations

        @return list of FeatureMutationStep (correponding to after first mutation to before last mutation)
        """
        feat_mutation_steps = []

        old_mutation_pos = None
        feat_dict_prev = dict()
        flanked_seq = seq_mut_order.get_seq_at_step(update_step_start, flanked=True)
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
            feat_mutation_steps.append(FeatureMutationStep(
                mutating_pos_feat,
                feat_dict_prev,
                feat_dict_curr,
            ))

            # Apply mutation
            flanked_seq = mutate_string(
                flanked_seq,
                mutation_pos + self.half_motif_len,
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
        @param flanked_seq: must be a FLANKED sequence
        @param already_mutated_pos: set of positions that already mutated - dont calculate feature vals for these
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
            first_mutation_pos + self.half_motif_len,
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

        return first_mut_pos_feat, FeatureMutationStep(
            second_mut_pos_feat,
            feat_dict_future,
            feat_dict_curr,
        )

    @profile
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

        @return FeatureMutationStep
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
                mutation_step - 1,
                seq_mut_order,
                intermediate_seq,
                already_mutated_pos,
            )

        # Calculate the features in these special positions for updating the next mutation step's risk group
        # Get the feature vectors for the positions that will be affected by current mutation
        if calc_future_dict:
            feat_dict_future = self._get_feature_dict_for_region(
                mutation_pos,
                mutation_step,
                seq_mut_order,
                intermediate_seq,
                already_mutated_pos,
            )
        return mutating_pos_feat, feat_dict_curr, feat_dict_future

    @profile
    def _get_feature_dict_for_region(
        self,
        mutation_pos,
        mutation_step,
        seq_mut_order,
        intermediate_seq,
        already_mutated_pos,
    ):
        feat_dict = dict()
        start_region_idx = max(mutation_pos - self.half_motif_len, 0)
        end_region_idx = min(mutation_pos + self.half_motif_len, seq_mut_order.obs_seq_mutation.seq_len - 1)
        update_positions = range(start_region_idx, mutation_pos) + range(mutation_pos + 1, end_region_idx + 1)
        for pos in update_positions:
            if pos not in already_mutated_pos:
                # Only update the positions that are in the risk group (the ones that haven't mutated yet)
                submotif = intermediate_seq[pos: pos + self.motif_len]
                feat_dict[pos] = self.motif_dict[submotif]
        return feat_dict

    @profile
    def _create_feature_vec_for_pos(self, pos, intermediate_seq, seq_len, left_flank, right_flank):
        """
        @param pos: central mutating position
        @param intermediate_seq: intermediate sequence to determine motif, flanks removed
        @param left flank: left flank nucleotide information
        @param right flank: right flank nucleotide information

        Create features for subsequence using information from flanks.
        """
        # if motif length is one then submotifs will be single nucleotides and position remains unchanged
        if pos < self.half_motif_len:
            submotif = left_flank[pos:] + intermediate_seq[:self.half_motif_len + pos + 1]
        elif pos >= seq_len - self.half_motif_len:
            submotif = intermediate_seq[pos - self.half_motif_len:] + right_flank[:pos + self.half_motif_len - seq_len + 1]
        else:
            submotif = intermediate_seq[pos - self.half_motif_len: pos + self.half_motif_len + 1]

        return self.motif_dict[submotif]

    @profile
    def _construct_region_at_steps(self, pos_begin, pos_end, seq_mut_order, mutation_step_prev, mutation_step_curr):
        """
        @param pos: central mutating position
        @param seq_mut_order: ImputedSequenceMutations
        @param mutation_step: if negative, that means no mutations have occured yet
        TODO: UPDATE COMMENT
        @return the feature index at position `pos` after the `mutation_step`-th mutation step
        """
        obs_seq_mutation = seq_mut_order.obs_seq_mutation
        seq_len = obs_seq_mutation.seq_len
        region = ""
        if pos_begin < self.half_motif_len:
            # Position is very close to beginning of sequence - copy over the flanks
            region = obs_seq_mutation.left_flank[pos_begin:]
            range_begin = 0
        else:
            range_begin = pos_begin - self.half_motif_len

        if pos_end >= seq_len - self.half_motif_len:
            range_end = seq_len
        else:
            range_end = pos_end + self.half_motif_len + 1

        regions = [region, region]
        for i in range(range_begin, range_end):
            start_nuc = obs_seq_mutation.start_seq[i]
            end_nuc = obs_seq_mutation.end_seq[i]
            pos_mut_step = seq_mut_order.mutation_order_all_pos[i]
            regions[0] += start_nuc if pos_mut_step > mutation_step_prev else end_nuc
            regions[1] += start_nuc if pos_mut_step > mutation_step_curr else end_nuc

        if pos_end >= seq_len - self.half_motif_len:
            # Position is very close to end of sequence - copy over the flanks
            suffix = obs_seq_mutation.right_flank[:pos_end + self.half_motif_len - seq_len + 1]
            regions[0] += suffix
            regions[1] += suffix
        return regions


    @profile
    def _get_feature_idx_for_pos_at_step(self, pos, seq_mut_order, mutation_step):
        """
        @param pos: central mutating position
        @param seq_mut_order: ImputedSequenceMutations
        @param mutation_step: if negative, that means no mutations have occured yet

        @return the feature index at position `pos` after the `mutation_step`-th mutation step
        """
        seq_len = seq_mut_order.obs_seq_mutation.seq_len
        submotif = ""
        if pos < self.half_motif_len:
            # Position is very close to beginning of sequence - copy over the flanks
            submotif = seq_mut_order.obs_seq_mutation.left_flank[pos:]
            pos_range = range(self.half_motif_len + pos + 1)
        elif pos >= seq_len - self.half_motif_len:
            # Position is very close to end of sequence - dont go past the end
            pos_range = range(pos - self.half_motif_len, seq_len)
        else:
            pos_range = range(pos - self.half_motif_len, pos + self.half_motif_len + 1)

        for i in pos_range:
            if seq_mut_order.mutation_order_all_pos[i] > mutation_step:
                submotif += seq_mut_order.obs_seq_mutation.start_seq[i]
            else:
                submotif += seq_mut_order.obs_seq_mutation.end_seq[i]

        if pos >= seq_len - self.half_motif_len:
            # Position is very close to end of sequence - copy over the flanks
            submotif += seq_mut_order.obs_seq_mutation.right_flank[:pos + self.half_motif_len - seq_len + 1]
        idx = self.motif_dict[submotif]
        return idx

    def get_motif_list(self):
        motif_list = itertools.product(*([NUCLEOTIDES] * self.motif_len))
        return ["".join(m) for m in motif_list]
