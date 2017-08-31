import sys
import csv
import numpy as np
PARTIS_PATH = './partis'
sys.path.insert(1, PARTIS_PATH + '/python')
import glutils

class GermlineMetadata:
    def __init__(self, seq_val, seq_name, seq_freq, seq_family):
        self.val = seq_val
        self.name = seq_name
        self.freq = seq_freq
        self.family = seq_family


class GermlineSimulatorPartis:
    """
    Generates germline genes using partis
    To use this file, please use dev branch on partis
    """

    GERMLINE_FOLDER = "./partis/data/germlines"

    def __init__(self, organism="mouse", output_dir="_output"):
        assert(organism in ["human", "mouse"])
        self.glfo = glutils.read_glfo(self.GERMLINE_FOLDER + "/"  + organism, "igh")
        self.output_dir = output_dir
        self.allele_freq_file = self.output_dir + "/allele_freq.csv"
        self.ighv_file = self.output_dir + "/igh/ighv.fasta"

    def generate_germline_sets(self, num_sets=1, n_genes_per_region="20:1:1", n_sim_alleles_per_gene="1,2:1:1", min_sim_allele_prevalence_freq=0.1):
        """
        @param num_sets: number of germline sets to create
        @param n_genes_per_region: number of genes to choose for each of the V, D, and J regions (colon separated list ordered like v:d:j)
        @param n_sim_alleles_per_gene: number of alleles to choose for each of these genes (colon-separated list of comma separated lists)
        @param min_sim_allele_prevalence_freq: minimum prevalence ratio between any two alleles in the germline set
        """
        germline_seqs_dict = dict()
        for set_i in range(num_sets):
            germline_seqs, germline_freqs = self._generate_germline_set(
                num_sets,
                n_genes_per_region,
                n_sim_alleles_per_gene,
                min_sim_allele_prevalence_freq,
            )
            for seq_key, seq_val in germline_seqs.iteritems():
                seq_key_all = "s%d-%s" % (set_i, seq_key)
                seq_gene = seq_key.split("*", 1)[0]
                print seq_gene
                seq_freq = germline_freqs[seq_key]
                germline_seqs_dict[seq_key_all] = GermlineMetadata(seq_val, seq_key_all, seq_freq/num_sets, seq_gene)

        return germline_seqs_dict

    def _generate_germline_set(self, num_sets=1, n_genes_per_region="20:1:1", n_sim_alleles_per_gene="1,2:1:1", min_sim_allele_prevalence_freq=0.1):
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
                allele_seq = lines[line_idx * 2 + 1]

                # Trim allele until multiple of 3 - randomly pick a reading frame
                mod_seq_len = len(allele_seq) % 3
                if mod_seq_len != 0:
                    offset = np.random.choice(mod_seq_len + 1)
                    if mod_seq_len != offset:
                        allele_seq = allele_seq[offset:-(mod_seq_len - offset)]
                    else:
                        allele_seq = allele_seq[offset:]

                germline_seqs[allele] = allele_seq

        return germline_seqs, germline_freqs
