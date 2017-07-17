import sys
import csv
import numpy as np
PARTIS_PATH = './partis'
sys.path.insert(1, PARTIS_PATH + '/python')
import glutils

class GermlineSimulator:
    """
    To use this file, please use dev branch on partis
    """

    GERMLINE_FOLDER = "./partis/data/germlines"

    def __init__(self, organism="human", output_dir="_output", seed=0):
        np.random.seed(seed)
        assert(organism in ["human", "mouse"])
        self.glfo = glutils.read_glfo(self.GERMLINE_FOLDER + "/"  + organism, "igh")
        self.output_dir = output_dir
        self.allele_freq_file = self.output_dir + "/allele_freq.csv"
        self.ighv_file = self.output_dir + "/igh/ighv.fasta"

    def generate_germline_set(self, n_genes_per_region="20:1:1", n_sim_alleles_per_gene="1,2:1,2:1,2", min_sim_allele_prevalence_freq=0.1):
        """
        @param n_genes_per_region: number of genes to choose for each of the V, D, and J regions (colon separated list ordered like v:d:j)
        @param n_sim_alleles_per_gene: number of alleles to choose for each of these genes (colon-separated list of comma separated lists)
        @param min_sim_allele_prevalence_freq: minimum prevalence ratio between any two alleles in the germline set
        """
        glutils.generate_germline_set(self.glfo, n_genes_per_region, n_sim_alleles_per_gene, min_sim_allele_prevalence_freq, self.allele_freq_file)
        glutils.write_glfo(self.output_dir, self.glfo)

        # Read allele prevalences
        germline_freqs = dict()
        with open(self.allele_freq_file, "r") as f:
            allele_reader = csv.reader(f)
            allele_reader.next()
            for row in allele_reader:
                if row[0].startswith("IGHV"):
                    germline_freqs[row[0]] = float(row[1])

        # Read the selected germline alleles
        germline_seqs = dict()
        with open(self.ighv_file, "r") as f:
            lines = f.read().splitlines()
            for line_idx in range(len(lines)/2):
                allele = lines[line_idx * 2].replace(">", "")
                germline_seqs[allele] = lines[line_idx * 2 + 1]

        return germline_seqs, germline_freqs
