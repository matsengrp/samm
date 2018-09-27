from generic_feature_generator import GenericFeatureGenerator
from itertools import izip

class PositionFeatureGenerator(GenericFeatureGenerator):
    """
    Feature generator for position-based model

    Default is each position is a separate feature
    """
    def __init__(
            self,
            breaks=[],
            labels=None,
            max_seq_len=500,
            model_truncation=None,
            feats_to_remove=None,
        ):
        """
        @param breaks: numeric vector with one or more break to divide sequence; cuts sequence into intervals [a,b)
            e.g., a sequence of length 10 and breaks [0, 1, 5, 10] yields three regions:
                [0,1) (i.e., just the first position),
                [1, 5),
                [5, 10) (latter half of sequence)
            defaults to each position being a separate feature, i.e., breaks=range(len(seq))
            position of sequence starts at beginning of *raw* sequence before end-processing
        @param labels: character vector of labels for each interval; must be same length as breaks minus one
            e.g., in the above breaks=[0, 1, 5, 10], if we pass labels=['A', 'B', 'A'] then [0,1) and [5,10) will have the
            same coefficient and be labeled 'A'.
            defaults to each region having a unique, numeric label ([0, 1, 2] in the above example)
        @param model_truncation: ModelTruncation object
        @param feats_to_remove: list of features to remove if a model has not been fit yet
        """
        self.max_seq_len = max_seq_len

        if not breaks:
            breaks = range(self.max_seq_len)

        breaks = sorted(breaks)
        if len(breaks) > len(set(breaks)):
            raise ValueError('Cut points must be unique')
        if min(breaks) < 0 or max(breaks) > max_seq_len:
            raise ValueError('Invalid cut point provided')

        if labels is None:
            labels = [idx for idx in range(len(breaks)-1)]

        if len(labels) != len(breaks) - 1:
            raise ValueError('Invalid labels provided')

        self.breaks = breaks
        self.labels = labels

        self.feats_to_remove = model_truncation.feats_to_remove if model_truncation is not None else []
        if feats_to_remove is not None:
            self.feats_to_remove += feats_to_remove

        self.update_feats_after_removing(self.feats_to_remove)

    def update_feats_after_removing(self, feats_to_remove=[]):
        """
        take existing MotifGenerator and update it with features to remove
        """
        label_dict = {lab: [] for lab in self.labels}
        for start, end, lab in izip(self.breaks, self.breaks[1:], self.labels):
            label_dict[lab].append((start, end))

        all_feature_info_list = [(lab, break_list) for lab, break_list in sorted(label_dict.iteritems())]
        self.feature_info_list = [feat_tuple for feat_tuple in all_feature_info_list if feat_tuple not in feats_to_remove]

        self.pos_dict = {i: None for i in range(self.max_seq_len)}
        for idx, (lab, break_list) in enumerate(self.feature_info_list):
            for start, end in break_list:
                for pos in range(start, end):
                    self.pos_dict[pos] = idx

        self.feature_vec_len = len(self.feature_info_list)

    def print_label_from_info(self, info):
        """
        Print what feature this corresponds to.

        @param info: an element of feature_info_list
        """
        lab, break_list = info
        print_str = "position: %s, " % lab
        print_str += "; ".join(["[%d, %d)" % (start, end) for start, end in break_list])

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
