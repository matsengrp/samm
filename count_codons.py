import pandas as pd
import pickle
import sys
import argparse
import numpy as np
import csv
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import *

AA_TABLE = {
    'ata':'I', 'atc':'I', 'att':'I', 'atg':'M',
    'aca':'T', 'acc':'T', 'acg':'T', 'act':'T',
    'aac':'N', 'aat':'N', 'aaa':'K', 'aag':'K',
    'agc':'S', 'agt':'S', 'aga':'R', 'agg':'R',
    'cta':'L', 'ctc':'L', 'ctg':'L', 'ctt':'L',
    'cca':'P', 'ccc':'P', 'ccg':'P', 'cct':'P',
    'cac':'H', 'cat':'H', 'caa':'Q', 'cag':'Q',
    'cga':'R', 'cgc':'R', 'cgg':'R', 'cgt':'R',
    'gta':'V', 'gtc':'V', 'gtg':'V', 'gtt':'V',
    'gca':'A', 'gcc':'A', 'gcg':'A', 'gct':'A',
    'gac':'D', 'gat':'D', 'gaa':'E', 'gag':'E',
    'gga':'G', 'ggc':'G', 'ggg':'G', 'ggt':'G',
    'tca':'S', 'tcc':'S', 'tcg':'S', 'tct':'S',
    'ttc':'F', 'ttt':'F', 'tta':'L', 'ttg':'L',
    'tac':'Y', 'tat':'Y', 'taa':'_', 'tag':'_',
    'tgc':'C', 'tgt':'C', 'tga':'_', 'tgg':'W',
}

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed',
        type=int,
        help='Random number generator seed for replicability',
        default=1)
    parser.add_argument('--input-mutated-template',
        type=str,
        help='CSV file to output for mutated sequences',
        default='_output/mutated.csv')
    parser.add_argument('--output-codon',
        default=str,
        help='codon output file')

    parser.set_defaults()
    args = parser.parse_args()
    return args

def translate(seq):
    """
    I copied this code from https://www.geeksforgeeks.org/dna-protein-python-3/
    """

    protein =""
    for i in range(0, len(seq), 3):
        if len(seq[i:i + 3]) < 3:
            break
        codon = seq[i:i + 3]
        protein+= AA_TABLE[codon]
    return protein

def count_mutations(naive_seq, mut_seqs):
    naive_protein = translate(naive_seq)
    prot_len = len(naive_protein)
    print("TOTAL NUM MUT SEQUENCES", len(mut_seqs))

    # Look at what mutations occurred
    codon_index = {}
    codon_cols = []
    for i, a in enumerate(set(AA_TABLE.keys())):
        codon_index[a] = i
        codon_cols.append(a)

    mutation_table = np.zeros((prot_len, len(codon_index)))
    for seq_idx, mut_seq in enumerate(mut_seqs):
        for pos_idx in range(prot_len):
            mut_codon = mut_seq[pos_idx * 3: (pos_idx + 1) * 3]
            mutation_table[pos_idx, codon_index[mut_codon]] += 1
    mutation_table = pd.DataFrame(
            mutation_table,
            columns=codon_cols)
    return mutation_table

def main(args=sys.argv[1:]):
    args = parse_args()
    np.random.seed(args.seed)

    tot_input_mut_counts = 0
    for input_mut in glob.glob(args.input_mutated_template):
        input_mut_counts = pd.read_csv(input_mut)
        tot_input_mut_counts += input_mut_counts

    tot_input_mut_counts.to_csv(args.output_codon)

    #fig = plt.imshow(mutation_table.values)
    #plt.colorbar(fig)
    #plt.tight_layout()
    #plt.savefig(args.output_codon.replace("csv", "png"))


if __name__ == "__main__":
    main(sys.argv[1:])
