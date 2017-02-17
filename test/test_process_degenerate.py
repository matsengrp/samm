import unittest

from common import *
from constants import *
from models import ObservedSequenceMutations

class Degenerate_TestCase(unittest.TestCase):
    def test_processing(self):
        """
        Test processing of degenerate bases
        """

        MOTIF_LEN = 5

        START_SEQ = "tagacgntac...a.....acnntcgggaaaacannn"
        END_SEQ =   "nnnttgntgc......c..acnntaggtaacaaannn"

        # what they should be processed as
        PROC_START = "acgntacnnacnntcgggaaaaca"
        PROC_END =   "ttgntgcnnacnntaggtaacaaa"

        # what they should be trimmed to
        TRIM_START = "tacnnacnntcggg"
        TRIM_FLANK_LEFT = "gn"
        TRIM_FLANK_RIGHT = "aa"
        TRIM_DICT = {1: 'g', 10: 'a', 13: 't'}

        # can we process correctly?
        start_processed, end_processed = trim_degenerates_and_collapse(START_SEQ, END_SEQ, MOTIF_LEN)
        self.assertEquals(PROC_START, start_processed)
        self.assertEquals(PROC_END, end_processed)

        # can we handle edge mutations?
        obs_mute = ObservedSequenceMutations(start_processed, end_processed, MOTIF_LEN)
        self.assertEquals(obs_mute.start_seq, TRIM_START)
        self.assertEquals(obs_mute.left_flank, TRIM_FLANK_LEFT)
        self.assertEquals(obs_mute.right_flank, TRIM_FLANK_RIGHT)
        self.assertEquals(obs_mute.mutation_pos_dict, TRIM_DICT)

