import unittest

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

        START_SEQ = "tagacgntac...a.....acnntcgggaaaacannn"
        END_SEQ =   "nnnatgntgc......c..acnntaggtaacaaannn"

        # what they should be processed as
        random.seed(1533)

        PROC_START = "acgntacnnacnntcgggaaaaca"
        PROC_END =   "atgntgcnnacnntaggtaacaaa"
        for match in re.compile('n').finditer(PROC_START):
            random_nuc = random.choice(NUCLEOTIDES)
            PROC_START = mutate_string(PROC_START, match.start(), random_nuc)
            PROC_END = mutate_string(PROC_END, match.start(), random_nuc)

        # what they should be trimmed to
        # the left flank starts at position 2 and sequence at position 4, right flank at position 18
        TRIM_START = PROC_START[4:18]
        TRIM_FLANK_LEFT = PROC_START[2:4]
        TRIM_FLANK_RIGHT = PROC_START[18:20]
        TRIM_DICT = {1: 'g', 10: 'a', 13: 't'}

        # can we process correctly?
        # use the same seed to get the same random nucleotides
        random.seed(1533)

        start_processed, end_processed = process_degenerates_and_impute_nucleotides(START_SEQ, END_SEQ, MOTIF_LEN)
        self.assertEquals(PROC_START, start_processed)
        self.assertEquals(PROC_END, end_processed)

        # can we handle edge mutations?
        obs_mute = ObservedSequenceMutations(start_processed, end_processed, MOTIF_LEN)
        self.assertEquals(obs_mute.start_seq, TRIM_START)
        self.assertEquals(obs_mute.left_flank, TRIM_FLANK_LEFT)
        self.assertEquals(obs_mute.right_flank, TRIM_FLANK_RIGHT)
        self.assertEquals(obs_mute.mutation_pos_dict, TRIM_DICT)

