import unittest

from common import *
from lauras_data_module import *

class MCMC_EM_TestCase(unittest.TestCase):
    def test_partis_fns(self):
        """
        Test partis input functions
        """

        print 'chain, class, nonproductive reads'
        for chain in ['h', 'k', 'l']:
            if chain=='h':
                # check IgG
                igclass = 'G'
                annotations, germlines = get_annotation_paths_from_lauras_data(LAURA_DATA_PATH, chain=chain, ig_class=igclass)
                genes, seqs = read_partis_annotations(annotations, inferred_gls=germlines, chain=chain)
                print chain, igclass, len(seqs)

                # check IgM
                igclass = 'M'
                annotations, germlines = get_annotation_paths_from_lauras_data(LAURA_DATA_PATH, chain=chain, ig_class=igclass)
                genes, seqs = read_partis_annotations(annotations, inferred_gls=germlines, chain=chain)
                print chain, igclass, len(seqs)
            else:
                # check light chain
                igclass = chain.upper()
                annotations, germlines = get_annotation_paths_from_lauras_data(LAURA_DATA_PATH, chain=chain)
                genes, seqs = read_partis_annotations(annotations, inferred_gls=germlines, chain=chain)
                print chain, igclass, len(seqs)

