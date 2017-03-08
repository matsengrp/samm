import unittest

from read_data import *
from matsen_grp_data import *
from constants import *

class Input_Data_TestCase(unittest.TestCase):
    def test_data_input_fns(self):
        """
        Test various data processing/input
        """

        temp_genes = '_output/genes.csv'
        temp_seqs = '_output/seqs.csv'
        motif_len = 3

        seqs = read_gene_seq_csv_data(INPUT_GENES, INPUT_SEQS, motif_len)
        print len(seqs)

        seqs = read_gene_seq_csv_data(INPUT_GENES, INPUT_SEQS, motif_len, sample='sample-random')
        print len(seqs)

        seqs = read_gene_seq_csv_data(INPUT_GENES, INPUT_SEQS, motif_len, sample='sample-highly-mutated')
        print len(seqs)

        write_data_after_imputing(temp_genes, temp_seqs, INPUT_GENES, INPUT_SEQS, motif_len, verbose=False)
        seqs = read_gene_seq_csv_data(temp_genes, temp_seqs, motif_len)
        print len(seqs)

