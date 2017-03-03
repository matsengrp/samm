import unittest

from read_data import *
from matsen_grp_data import *
from constants import *

class Input_Data_TestCase(unittest.TestCase):
    def test_data_input_fns(self):
        """
        Test various data processing/input
        """

        motif_len = 3
        temp_genes = '_output/genes.csv'
        temp_seqs = '_output/seqs.csv'

        seqs = read_gene_seq_csv_data(INPUT_GENES, INPUT_SEQS, motif_len)
        print len(seqs)

        seqs = read_gene_seq_csv_data(INPUT_GENES, INPUT_SEQS, motif_len, sample_or_impute='impute-ancestors')
        print len(seqs)

        seqs = read_gene_seq_csv_data(INPUT_GENES, INPUT_SEQS, motif_len, sample_or_impute='sample-random')
        print len(seqs)

        seqs = read_gene_seq_csv_data(INPUT_GENES, INPUT_SEQS, motif_len, sample_or_impute='sample-highly-mutated')
        print len(seqs)

