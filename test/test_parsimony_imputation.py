import unittest

from read_data import *
from matsen_grp_data import *
from constants import *
from submotif_feature_generator import SubmotifFeatureGenerator

class Input_Data_TestCase(unittest.TestCase):
    def test_data_input_fns(self):
        """
        Test various data processing/input
        """

        temp_genes = '_output/genes.csv'
        temp_seqs = '_output/seqs.csv'
        motif_len = 3

        seqs, metadata = read_gene_seq_csv_data(INPUT_GENES, INPUT_SEQS, motif_len)
        print len(seqs)

        seqs, metadata = read_gene_seq_csv_data(INPUT_GENES, INPUT_SEQS, motif_len, sample=2)
        print len(seqs)

        seqs, metadata = read_gene_seq_csv_data(INPUT_GENES, INPUT_SEQS, motif_len, sample=3)
        print len(seqs)

        write_data_after_imputing(temp_genes, temp_seqs, INPUT_GENES, INPUT_SEQS, motif_len, verbose=False)
        seqs, metadata = read_gene_seq_csv_data(temp_genes, temp_seqs, motif_len)
        print len(seqs)

    def test_statistics(self):

        motif_len = 3

        feat_generator = SubmotifFeatureGenerator(motif_len=motif_len)
        seqs, metadata = read_gene_seq_csv_data(INPUT_GENES, INPUT_SEQS, motif_len)
        print get_data_statistics_print_lines(seqs, feat_generator)

