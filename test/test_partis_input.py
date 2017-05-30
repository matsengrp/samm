import unittest

from read_data import write_partis_data_from_annotations, read_gene_seq_csv_data
from matsen_grp_data import CUI_DATA_PATH

class Input_Data_TestCase(unittest.TestCase):
    def test_partis_fns(self):
        """
        Test partis input functions
        """

        temp_genes = '_output/genes.csv'
        temp_seqs = '_output/seqs.csv'
        motif_len = 3

        for dataset in [CUI_DATA_PATH]:
            write_partis_data_from_annotations(temp_genes, temp_seqs, dataset, dataset+'/meta.csv')
            for chain in ['igk', 'igl']:
                seqs, meta = read_gene_seq_csv_data(temp_genes, temp_seqs, motif_len, subset_cols=['locus'], subset_vals=[chain])
                print chain, len(seqs), len(set([elt['subject'] for elt in meta]))

