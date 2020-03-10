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
    parser.add_argument('--input-naive',
        type=str,
        help='CSV file to output for naive sequences',
        default='_output/naive.csv')
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


def main(args=sys.argv[1:]):
    args = parse_args()
    np.random.seed(args.seed)

    # Read naive protein
    with open(args.input_naive, "r") as f:
        seqreader = csv.DictReader(f, delimiter=',')
        for row in seqreader:
            naive_protein = translate(row["germline_sequence"])
            assert "_" not in naive_protein

    mut_proteins = []
    for input_mut in glob.glob(args.input_mutated_template):
        print(input_mut)
        # Read mutated protiens
        with open(input_mut, "r") as f:
            seqreader = csv.DictReader(f, delimiter=',')
            for row in seqreader:
                mut_protein = translate(row["sequence"])
                mut_proteins.append(mut_protein)
    print("TOTAL NUM MUT SEQUENCES", len(mut_proteins))

    # Look at what mutations occurred
    aa_index = {}
    aa_cols = []
    for i, a in enumerate(set(AA_TABLE.values())):
        aa_index[a] = i
        aa_cols.append(a)

    mutation_table = np.zeros((len(naive_protein), len(aa_index)))
    for prot_idx, mut_protein in enumerate(mut_proteins):
        #print("protein", prot_idx)
        #if "_" in mut_protein:
        #    print("early stop codon")

        for pos_idx, (naive_aa, mut_aa) in enumerate(zip(naive_protein, mut_protein)):
            if naive_aa != mut_aa:
                #print(pos_idx, "%s -> %s" % (naive_aa, mut_aa))
                mutation_table[pos_idx, aa_index[mut_aa]] += 1
    mutation_table = pd.DataFrame(
            mutation_table,
            columns=aa_cols)
    mutation_table.to_csv(args.output_codon)

    fig = plt.imshow(mutation_table.values)
    plt.colorbar(fig)
    plt.tight_layout()
    plt.savefig(args.output_codon.replace("csv", "png"))


if __name__ == "__main__":
    main(sys.argv[1:])
