import scipy.sparse

from common import mutate_string
from feature_generator import FeatureGenerator, MultiFeatureMutationStep

class GenericFeatureGenerator(FeatureGenerator):
    """
    Subclass this to have various types of feature vector generators.
    We assume all feature vectors are composed of ones and zeros.
    Each feature vector will be a dictionary keyed on some property of the sequence being the
    corresponding index in the list of properties.
    We call these "sparse feature vectors".
    """

    def get_base_features(self, obs_seq_mutation):
        """
        Create the feature matrices and feature vector dictionary
        before any mutations have occurred

        @param obs_seq_mutation: ObservedSequenceMutations

        @return sparse matrix with presence/absence of features per position for this sequence
        """
        indices = []
        start_idx = 0
        indptr = [start_idx]

        for pos in range(obs_seq_mutation.seq_len):
            feat_idx = self._get_mutating_pos_feat_idx(pos, obs_seq_mutation.start_seq_with_flanks)
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

    def create_for_mutation_steps(
        self,
        seq_mut_order,
        left_update_region,
        right_update_region,
    ):
        """
        Calculate the feature values for the mutation steps
        Only returns the deltas at each mutation step

        @param seq_mut_order: ImputedSequenceMutations

        @return list of FeatureMutationStep (correponding to after first mutation to before last mutation)
        """
        feat_mutation_steps = []

        old_mutation_pos = None
        intermediate_seq = seq_mut_order.obs_seq_mutation.start_seq_with_flanks

        feat_dict_prev = dict()
        already_mutated_pos = set()
        for mutation_step, mutation_pos in enumerate(seq_mut_order.mutation_order):
            feat_dict_curr, feat_dict_future = self.update_mutation_step(
                mutation_step,
                mutation_pos,
                old_mutation_pos,
                seq_mut_order,
                intermediate_seq,
                already_mutated_pos,
                left_update_region=left_update_region,
                right_update_region=right_update_region,
            )
            mutating_pos_feat_idx = self._get_mutating_pos_feat_idx(mutation_pos, intermediate_seq)
            feat_mutation_steps.append(MultiFeatureMutationStep(
                mutating_pos_feat_idx,
                mutation_pos,
                neighbors_feat_old=feat_dict_prev,
                neighbors_feat_new=feat_dict_curr,
            ))

            # Apply mutation
            curr_mutation_pos = mutation_pos + seq_mut_order.obs_seq_mutation.left_flank_len
            intermediate_seq = mutate_string(
                intermediate_seq,
                curr_mutation_pos,
                seq_mut_order.obs_seq_mutation.end_seq_with_flanks[curr_mutation_pos],
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
        left_update_region,
        right_update_region,
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

        already_mutated_pos = set(seq_mut_order.mutation_order[:update_step_start])
        for mutation_step in range(update_step_start, seq_mut_order.obs_seq_mutation.num_mutations):
            mutation_pos = seq_mut_order.mutation_order[mutation_step]
            feat_dict_curr, feat_dict_future = self.update_mutation_step(
                mutation_step,
                mutation_pos,
                old_mutation_pos,
                seq_mut_order,
                flanked_seq,
                already_mutated_pos,
                left_update_region=left_update_region,
                right_update_region=right_update_region,
            )
            mutating_pos_feat_idx = self._get_mutating_pos_feat_idx(mutation_pos, flanked_seq)
            feat_mutation_steps.append(MultiFeatureMutationStep(
                mutating_pos_feat_idx,
                mutation_pos,
                neighbors_feat_old=feat_dict_prev,
                neighbors_feat_new=feat_dict_curr,
            ))

            # Apply mutation
            curr_mutation_pos = mutation_pos + seq_mut_order.obs_seq_mutation.left_flank_len
            flanked_seq = mutate_string(
                flanked_seq,
                curr_mutation_pos,
                seq_mut_order.obs_seq_mutation.end_seq_with_flanks[curr_mutation_pos],
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
        left_update_region,
        right_update_region,
    ):
        """
        @param seq_mut_order: a list of the positions in the mutation order
        @param update_step: the index of the mutation step being shuffled with the (`update_step` + 1)-th step
        @param flanked_seq: must be a FLANKED sequence
        @param already_mutated_pos: set of positions that already mutated - dont calculate feature vals for these

        @return a tuple with the feature index at this mutation step and the feature mutation step of the next mutation step
        """
        feat_mutation_steps = []
        first_mutation_pos = seq_mut_order.mutation_order[update_step]
        second_mutation_pos = seq_mut_order.mutation_order[update_step + 1]

        _, feat_dict_future = self.update_mutation_step(
            update_step,
            first_mutation_pos,
            None,
            seq_mut_order,
            flanked_seq,
            already_mutated_pos,
            left_update_region=left_update_region,
            right_update_region=right_update_region,
        )
        first_mut_pos_feat_idx = self._get_mutating_pos_feat_idx(first_mutation_pos, flanked_seq)

        # Apply mutation
        curr_mutation_pos = first_mutation_pos + seq_mut_order.obs_seq_mutation.left_flank_len
        flanked_seq = mutate_string(
            flanked_seq,
            curr_mutation_pos,
            seq_mut_order.obs_seq_mutation.end_seq_with_flanks[curr_mutation_pos],
        )

        feat_dict_curr, _ = self.update_mutation_step(
            update_step + 1,
            second_mutation_pos,
            first_mutation_pos,
            seq_mut_order,
            flanked_seq,
            already_mutated_pos,
            calc_future_dict=False,
            left_update_region=left_update_region,
            right_update_region=right_update_region,
        )
        second_mut_pos_feat_idx = self._get_mutating_pos_feat_idx(second_mutation_pos, flanked_seq)

        return first_mut_pos_feat_idx, MultiFeatureMutationStep(
            second_mut_pos_feat_idx,
            second_mutation_pos,
            neighbors_feat_old=feat_dict_future,
            neighbors_feat_new=feat_dict_curr,
        )

    def update_mutation_step(
            self,
            mutation_step,
            mutation_pos,
            old_mutation_pos,
            seq_mut_order,
            intermediate_seq,
            already_mutated_pos,
            calc_future_dict=True,
            left_update_region=0,
            right_update_region=0,
        ):
        """
        Does the heavy lifting for calculating feature vectors at a given mutation step

        @param mutation_step: mutation step index
        @param mutation_pos: the position that is mutating
        @param old_mutation_pos: the position that mutated previously - None if this is first mutation
        @param seq_mut_order: ImputedSequenceMutations
        @param intermediate_seq: nucleotide sequence INCLUDING flanks - before the mutation step occurs
        @param already_mutated_pos: list of positions that have already mutated
        @param calc_future_dict: calculate feat_dict_future (dict with positions next to current mutation)
        @param left_update_region: from CombinedFeatureGenerator
        @param right_update_region: from CombinedFeatureGenerator

        @return tuple with
            1. a dict with the positions next to the previous mutation and their feature index
            2. a dict with the positions next to the current mutation and their feature index
        """
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
                left_update_region,
                right_update_region,
            )

        # Calculate the features in these special positions for updating the next mutation step's risk group
        # Get the feature vectors for the positions that will be affected by current mutation
        if calc_future_dict:
            feat_dict_future = self._get_feature_dict_for_region(
                mutation_pos,
                intermediate_seq,
                seq_mut_order.obs_seq_mutation.seq_len,
                already_mutated_pos,
                left_update_region,
                right_update_region,
            )
        return feat_dict_curr, feat_dict_future

    def create_for_sequence(self, seq_str, left_flank, right_flank, do_feat_vec_pos=None):
        feat_vec_dict = dict()
        seq_len = len(seq_str)
        if do_feat_vec_pos is None:
            do_feat_vec_pos = range(len(seq_str))
        # only generate feature vectors for positions in do_feat_vec_pos---others not in the risk group
        for pos in do_feat_vec_pos:
            feat_vec_dict[pos] = self._get_mutating_pos_feat_idx(pos, left_flank + seq_str + right_flank)
        return feat_vec_dict

    def _get_feature_dict_for_region(
        self,
        position,
        intermediate_seq,
        seq_len,
        already_mutated_pos,
        left_update_region,
        right_update_region,
    ):
        """
        @param position: the position around which to calculate the feature indices for
        @param intermediate_seq: the nucleotide sequence
        @param seq_len: the length of this sequence
        @param already_mutated_pos: which positions already mutated - dont calculate features for these positions

        @return a dict with the positions next to the given position and their feature index
        """
        feat_dict = dict()
        start_region_idx = max(position - left_update_region, 0)
        end_region_idx = min(position + right_update_region, seq_len - 1)
        update_positions = range(start_region_idx, position) + range(position + 1, end_region_idx + 1)
        for pos in update_positions:
            if pos not in already_mutated_pos:
                # Only update the positions that are in the risk group (the ones that haven't mutated yet)
                feat_dict[pos] = self._get_mutating_pos_feat_idx(pos, intermediate_seq)
        return feat_dict

    def _get_mutating_pos_feat_idx(self, pos, seq_with_flanks):
        """
        Feature generator--specific function.
        Take sequence and mutating position and return the index of the feature vector.

        @param pos: mutating position
        @param seq_with_flanks: nucleotide sequence with flanks

        @return index of feature vector for this mutating position
        """
        raise NotImplementedError()
