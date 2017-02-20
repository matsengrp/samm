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
            feat_vect = self._create_feature_vec_for_pos(pos, obs_seq_mutation.start_seq, obs_seq_mutation.seq_len, obs_seq_mutation.left_flank, obs_seq_mutation.right_flank)
            feat_dict[pos] = feat_vect
            indptr.append(pos + 1)
            indices.append(feat_vect)

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
        for mutation_step, mutation_pos in enumerate(seq_mut_order.mutation_order[:-1]):
            # print "mutation_step", mutation_step
            feat_mut_step = self._update_mutation_step(
                mutation_step,
                mutation_pos,
                old_mutation_pos,
                seq_mut_order,
            )
            feat_mutation_steps.append(feat_mut_step)
            old_mutation_pos = mutation_pos

        # print "mutation_step", seq_mut_order.obs_seq_mutation.num_mutations - 1
        final_mutation_feat = self._get_feature_idx_for_pos_at_step(
            seq_mut_order.mutation_order[-1],
            seq_mut_order,
            seq_mut_order.obs_seq_mutation.num_mutations - 1,
        )
        feat_mutation_steps.append(FeatureMutationStep(final_mutation_feat))

        return feat_mutation_steps

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
    @profile
    def _update_mutation_step(
            self,
            mutation_step,
            mutation_pos,
            old_mutation_pos,
            seq_mut_order,
        ):
        """
        Does the heavy lifting for calculating feature vectors at a given mutation step
        @param mutation_step: mutation step index
        @param mutation_pos: the position that is mutating
        @param old_mutation_pos: the position that mutated previously - None if this is first mutation
        @param seq_mut_order: ImputedSequenceMutations

        @return FeatureMutationStep
        """
        # print "seq_mut_order.obs_seq_mutation", seq_mut_order.obs_seq_mutation.start_seq
        # print "mutation_pos", mutation_pos

        mutating_pos_feat = self._get_feature_idx_for_pos_at_step(mutation_pos, seq_mut_order, mutation_step - 1)
        feat_vec_dict_new = dict()
        feat_vec_dict_old = dict()
        if old_mutation_pos is not None:
            start_region_idx = max(old_mutation_pos - self.half_motif_len, 0)
            end_region_idx = min(old_mutation_pos + self.half_motif_len, seq_mut_order.obs_seq_mutation.seq_len - 1)
            # print "old_mutation_pos", old_mutation_pos
            # print "start_region_idx", start_region_idx, "end", end_region_idx
            seq_at_step = self._construct_region_at_step(
                start_region_idx,
                end_region_idx,
                seq_mut_order,
                mutation_step - 1
            )
            # print "seq_at_step", seq_at_step
            seq_at_prev_step = self._construct_region_at_step(
                start_region_idx,
                end_region_idx,
                seq_mut_order,
                mutation_step - 2
            )
            # print "seq_at_prev_step", seq_at_prev_step
            # Calculate features for positions in the risk group at the time of this mutation step
            # Only requires updating feature values that were close to the previous mutation

            # Get the feature vectors for the positions that might be affected by the previous mutation
            update_positions = range(start_region_idx, old_mutation_pos) + range(old_mutation_pos + 1, end_region_idx + 1)
            # print "updated_positions", update_positions

            for pos in update_positions:
                if seq_mut_order.mutation_order_all_pos[pos] > mutation_step - 1:
                    # Only update the positions that are in the risk group (the ones that haven't mutated yet)
                    feat_vec_dict_new[pos] = self._get_feature_idx(pos - start_region_idx + self.half_motif_len, seq_at_step)
                    feat_vec_dict_old[pos] = self._get_feature_idx(pos - start_region_idx + self.half_motif_len, seq_at_prev_step)

        return FeatureMutationStep(
            mutating_pos_feat,
            feat_vec_dict_new,
            feat_vec_dict_old
        )

    def _get_feature_idx(self, pos, seq):
        # print "seq", seq
        # print "pos,", pos
        # print "pos - self.half_motif_len", pos - self.half_motif_len
        # print "pos + self.half_motif_len + 1", pos + self.half_motif_len + 1
        submotif = seq[pos - self.half_motif_len: pos + self.half_motif_len + 1]
        # print "submotif", submotif
        return self.motif_dict[submotif]

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

        # generate random nucleotide if an "n" occurs in the middle of a sequence
        if 'n' in submotif:
            for match in re.compile('n').finditer(submotif):
                submotif = submotif[:match.start()] + random.choice(NUCLEOTIDES) + submotif[(match.start()+1):]

        idx = self.motif_dict[submotif]
        return idx

    @profile
    def _construct_region_at_step(self, pos_begin, pos_end, seq_mut_order, mutation_step):
        """
        @param pos: central mutating position
        @param seq_mut_order: ImputedSequenceMutations
        @param mutation_step: if negative, that means no mutations have occured yet

        @return the feature index at position `pos` after the `mutation_step`-th mutation step
        """
        seq_len = seq_mut_order.obs_seq_mutation.seq_len
        region = ""
        if pos_begin < self.half_motif_len:
            # Position is very close to beginning of sequence - copy over the flanks
            region = seq_mut_order.obs_seq_mutation.left_flank[pos_begin:]
            range_begin = 0
        else:
            range_begin = pos_begin - self.half_motif_len

        if pos_end >= seq_len - self.half_motif_len:
            range_end = seq_len
        else:
            range_end = pos_end + self.half_motif_len + 1

        for i in range(range_begin, range_end):
            if seq_mut_order.mutation_order_all_pos[i] > mutation_step:
                region += seq_mut_order.obs_seq_mutation.start_seq[i]
            else:
                region += seq_mut_order.obs_seq_mutation.end_seq[i]

        if pos_end >= seq_len - self.half_motif_len:
            # Position is very close to end of sequence - copy over the flanks
            region += seq_mut_order.obs_seq_mutation.right_flank[:pos_end + self.half_motif_len - seq_len + 1]

        return region


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
