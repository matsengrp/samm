from generic_feature_generator import GenericFeatureGenerator

class PositionFeatureGenerator(GenericFeatureGenerator):
    """
    Feature generator for position-based model

    Default is each position is a separate feature
    """
    def __init__(
            self,
            breaks=[],
            max_seq_len=500,
            model_truncation=None,
            feats_to_remove=[],
        ):
        """
        @param breaks: numeric vector with one or more break to divide sequence; cuts sequence into intervals [a,b)
            e.g., a sequence of length 10 and breaks [1, 5] yields three regions:
                [0,1) (i.e., just the first position),
                [1, 5),
                [5, 10) (latter half of sequence)
            defaults to each position being a separate feature, i.e., breaks=range(len(seq))
            position of sequence starts at beginning of *raw* sequence before end-processing
        @param model_truncation: ModelTruncation object
        @param feats_to_remove: list of features to remove if a model has not been fit yet
        """
        self.max_seq_len = max_seq_len

        breaks = self._process_breaks(breaks)
        if not breaks:
            breaks = range(self.max_seq_len)

        self.breaks = breaks

        self.feats_to_remove = model_truncation.feats_to_remove if model_truncation is not None else []
        if feats_to_remove is not None:
            self.feats_to_remove += feats_to_remove

        self.update_feats_after_removing(self.feats_to_remove)

    def _process_breaks(self, breaks):
        """
        process break vector
        """
        processed_breaks = []
        for brk in sorted(list(set(breaks))):
            if brk < 0:
                continue
            elif brk >= self.max_seq_len:
                processed_breaks.append(self.max_seq_len)
                break
            processed_breaks.append(brk)

        return processed_breaks

    def update_feats_after_removing(self, feats_to_remove=[]):
        """
        take existing MotifGenerator and update it with features to remove
        """

        all_feature_info_list = [('position', start, end) for start, end in zip(self.breaks, self.breaks[1:])]
        self.feature_info_list = [feat_tuple for feat_tuple in all_feature_info_list if feat_tuple not in feats_to_remove]

        self.pos_dict = {i: None for i in range(self.max_seq_len)}
        for idx, (lab, start, end) in enumerate(all_feature_info_list):
            if (lab, start, end) in feats_to_remove:
                continue
            for pos in range(self.max_seq_len):
                if pos in range(start, end):
                    self.pos_dict[pos] = idx

        self.feature_vec_len = len(self.feature_info_list)

    def print_label_from_info(self, info):
        """
        Print what feature this corresponds to.

        @param info: an element of feature_info_list
        """
        _, start, end = info
        if start == 0:
            print_str = "position: [start, %d)" % end
        elif end == self.max_seq_len:
            print_str = "position: [%d, end)" % start
        else:
            print_str = "position: [%d, %d)" % (start, end)
        return print_str

    def _get_mutating_pos_feat_idx(self, pos, seq_with_flanks, obs_seq_mutation):
        """
        The fact that this module takes a flanked sequence means that some of the submotif length information is contained
        in seq_with_flanks, and we need to use flank_len_offset to get the correct feature.

        @param pos: mutating position
        @param seq_with_flanks: sequence to determine motif, flanks included

        @return index of feature vector for this mutating position
        """
        return self.pos_dict[obs_seq_mutation.raw_pos[pos]]
