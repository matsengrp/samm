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

        START_SEQ = "acgntac...a.....acnntcgggaaaaca"
        END_SEQ =   "ttgntgc......c..acnntaggtaacaaa"

        # what they should be processed as
        PROC_START = "acgntacnnacnntcgggaaaaca"
        PROC_END =   "ttgntgcnnacnntaggtaacaaa"

        # what they should be trimmed to
        TRIM_START = "gntacnnacnntcgggaa"
        TRIM_DICT = {3: 'g', 12: 'a', 15: 't'}

        # can we process correctly?
        start_processed, end_processed = process_degenerates(START_SEQ, END_SEQ, MOTIF_LEN)
        self.assertEquals(PROC_START, start_processed)
        self.assertEquals(PROC_END, end_processed)

        # can we handle edge mutations?
        obs_mute = ObservedSequenceMutations(start_processed, end_processed, MOTIF_LEN)
        self.assertEquals(obs_mute.start_seq, TRIM_START)
        self.assertEquals(obs_mute.mutation_pos_dict, TRIM_DICT)
