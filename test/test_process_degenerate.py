import unittest
import numpy as np

from common import *
from constants import *
from models import ObservedSequenceMutations
import random

class Degenerate_TestCase(unittest.TestCase):
    def test_processing(self):
        """
        Test processing of degenerate bases
        """

        MOTIF_LEN = 5
        MAX_FLANK_LEN = MOTIF_LEN/2

        START_SEQ = "tagacgntac...a.....acnntcgggaaaacannn"
        END_SEQ =   "nnnatgntgc......c..acnntaggtaacaaannn"

        random.seed(1533)

        # what they should be processed as
        PROC_START = "nnnacgntacnnacnntcgggaaaacannn"
        PROC_END =   "nnnatgntgcnnacnntaggtaacaaannn"
        COLLAPSE_LIST = [(MAX_FLANK_LEN, 7, 16)]

        # generate random nucleotide if an "n" occurs in the middle of a sequence
        for idx in [6, 10, 11, 14, 15]:
            random_nuc = random.choice(NUCLEOTIDES)
            PROC_START = mutate_string(PROC_START, idx, random_nuc)
            PROC_END = mutate_string(PROC_END, idx, random_nuc)

        # what they should be trimmed to
        # the left flank starts at position 2 and sequence at position 4, right flank at position 18
        TRIM_START = PROC_START[7:21]
        TRIM_FLANK_LEFT = PROC_START[5:7]
        TRIM_FLANK_RIGHT = PROC_START[21:23]
        TRIM_DICT = {1: 'g', 10: 'a', 13: 't'}

        # can we process correctly?
        # use the same seed to get the same random nucleotides
        random.seed(1533)

        start_processed, end_processed, collapse_list = process_degenerates_and_impute_nucleotides(START_SEQ, END_SEQ, MAX_FLANK_LEN)
        self.assertEquals(PROC_START, start_processed)
        self.assertEquals(PROC_END, end_processed)
        self.assertEquals(COLLAPSE_LIST, collapse_list)

        # can we handle edge mutations?
        obs_mute = ObservedSequenceMutations(start_processed, end_processed, MOTIF_LEN, collapse_list=collapse_list)
        self.assertEquals(obs_mute.start_seq, TRIM_START)
        self.assertEquals(obs_mute.left_flank, TRIM_FLANK_LEFT)
        self.assertEquals(obs_mute.right_flank, TRIM_FLANK_RIGHT)
        self.assertEquals(obs_mute.mutation_pos_dict, TRIM_DICT)
        self.assertEquals(obs_mute.raw_pos[0], 7)
        self.assertEquals(obs_mute.raw_pos[9], 23)

        self.assertEquals(len(_pad_vec_with_nan(obs_mute, np.array(list(obs_mute.start_seq)))), len(START_SEQ))

# Testing padding in mutation order gibbs
def _pad_vec_with_nan(obs_mute, vec):
    """
    Pad vector with NaN values both for the flanking positions and for the interior "n" values that were collapsed during processing
    """
    padded_vec = np.concatenate((
        [np.nan] * (obs_mute.left_position_offset),
        vec,
        [np.nan] * (obs_mute.right_position_offset),
    ))
    # insert NaNs where we had collapsed in processing the data before
    for half_motif_len, string_start, string_end in sorted(obs_mute.collapse_list, key=lambda val: val[1]):
        start_idx = obs_mute.left_position_offset + string_start - half_motif_len
        if start_idx <= len(padded_vec):
            # if it's greater then we'll pad the end regardless
            to_insert = [np.nan] * (string_end - string_start - half_motif_len)
            padded_vec = np.insert(padded_vec, start_idx, to_insert)
    return padded_vec
