import unittest

from read_data import *
from matsen_grp_data import *

class Input_Data_TestCase(unittest.TestCase):
    def test_partis_fns(self):
        """
        Test partis input functions
        """

        temp_genes = '_output/genes.csv'
        temp_seqs = '_output/seqs.csv'
        motif_len = 3

        for dataset in [LAURA_DATA_PATH, KATE_DATA_PATH]:
            print 'dataset: ', dataset
            print 'chain, class, nonproductive reads'
            for chain in ['h', 'k', 'l']:
                if chain=='h':
                    # check IgG
                    igclass = 'G'
                    annotations, germlines = get_paths_to_partis_annotations(dataset, chain=chain, ig_class=igclass)
                    write_partis_data_from_annotations(temp_genes, temp_seqs, annotations, inferred_gls=germlines, chain=chain)
                    genes, seqs = read_gene_seq_csv_data(temp_genes, temp_seqs, motif_len)
                    print chain, igclass, len(seqs)

                    # check IgM
                    igclass = 'M'
                    annotations, germlines = get_paths_to_partis_annotations(dataset, chain=chain, ig_class=igclass)
                    write_partis_data_from_annotations(temp_genes, temp_seqs, annotations, inferred_gls=germlines, chain=chain)
                    genes, seqs = read_gene_seq_csv_data(temp_genes, temp_seqs, motif_len)
                    print chain, igclass, len(seqs)
                else:
                    # check light chain
                    igclass = chain.upper()
                    annotations, germlines = get_paths_to_partis_annotations(dataset, chain=chain)
                    write_partis_data_from_annotations(temp_genes, temp_seqs, annotations, inferred_gls=germlines, chain=chain)
                    genes, seqs = read_gene_seq_csv_data(temp_genes, temp_seqs, motif_len)
                    print chain, igclass, len(seqs)
