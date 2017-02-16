import unittest

from common import *
from constants import *
from models import ObservedSequenceMutations

class Degenerate_TestCase(unittest.TestCase):
    def test_processing(self):
        """
        Test processing of degenerate bases
        """
        START_SEQ = "acgntac...a.....acnntcgggaaaaca"
        END_SEQ =   "ttgntgc......c..acnntaggtaaaaaa"

        for motif_len in [3, 5]:
            print "motif length: ", motif_len

            # can we process correctly?
            start_processed, end_processed = process_degenerates(START_SEQ, END_SEQ, motif_len)
            print start_processed
            print end_processed

            # can we handle edge mutations?
            obs_mute = ObservedSequenceMutations(start_processed, end_processed, motif_len)
            print obs_mute

            # can we read real data?
            gene_dict, obs_data_raw = read_gene_seq_csv_data(INPUT_GENES, INPUT_SEQS, motif_len)

