import unittest

from common import *
from matsen_grp_data import *

class Input_Data_TestCase(unittest.TestCase):
    def test_partis_fns(self):
        """
        Test partis input functions
        """

        for dataset in [LAURA_DATA_PATH, KATE_DATA_PATH]:
            print 'dataset: ', dataset
            print 'chain, class, nonproductive reads'
            for chain in ['h', 'k', 'l']:
                if chain=='h':
                    # check IgG
                    igclass = 'G'
                    annotations, germlines = get_paths_to_partis_annotations(dataset, chain=chain, ig_class=igclass)
                    genes, seqs = read_partis_annotations(annotations, inferred_gls=germlines, chain=chain)
                    print chain, igclass, len(seqs)
    
                    # check IgM
                    igclass = 'M'
                    annotations, germlines = get_paths_to_partis_annotations(dataset, chain=chain, ig_class=igclass)
                    genes, seqs = read_partis_annotations(annotations, inferred_gls=germlines, chain=chain)
                    print chain, igclass, len(seqs)
                else:
                    # check light chain
                    igclass = chain.upper()
                    annotations, germlines = get_paths_to_partis_annotations(dataset, chain=chain)
                    genes, seqs = read_partis_annotations(annotations, inferred_gls=germlines, chain=chain)
                    print chain, igclass, len(seqs)

