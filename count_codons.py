import pickle
import sys
import argparse
import numpy as np
import os
import os.path
import csv
import subprocess

from common import *

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed',
        type=int,
        help='Random number generator seed for replicability',
        default=1)
    parser.add_argument('--input-mutated',
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

    table = {
        'ata':'i', 'atc':'i', 'att':'i', 'atg':'m',
        'aca':'t', 'acc':'t', 'acg':'t', 'act':'t',
        'aac':'n', 'aat':'n', 'aaa':'k', 'aag':'k',
        'agc':'s', 'agt':'s', 'aga':'r', 'agg':'r',
        'cta':'l', 'ctc':'l', 'ctg':'l', 'ctt':'l',
        'cca':'p', 'ccc':'p', 'ccg':'p', 'cct':'p',
        'cac':'h', 'cat':'h', 'caa':'q', 'cag':'q',
        'cga':'r', 'cgc':'r', 'cgg':'r', 'cgt':'r',
        'gta':'v', 'gtc':'v', 'gtg':'v', 'gtt':'v',
        'gca':'a', 'gcc':'a', 'gcg':'a', 'gct':'a',
        'gac':'d', 'gat':'d', 'gaa':'e', 'gag':'e',
        'gga':'g', 'ggc':'g', 'ggg':'g', 'ggt':'g',
        'tca':'s', 'tcc':'s', 'tcg':'s', 'tct':'s',
        'ttc':'f', 'ttt':'f', 'tta':'l', 'ttg':'l',
        'tac':'y', 'tat':'y', 'taa':'_', 'tag':'_',
        'tgc':'c', 'tgt':'c', 'tga':'_', 'tgg':'w',
    }
    protein =""
    print("MOD 3", len(seq) % 3)
    assert len(seq) % 3 == 0
    for i in range(0, len(seq), 3):
        codon = seq[i:i + 3]
        print(codon)
        protein+= table[codon]
    return protein


def main(args=sys.argv[1:]):
    args = parse_args()
    np.random.seed(args.seed)

    with open(args.input_naive, "r") as f:
        seqreader = csv.DictReader(f, delimiter=',')
        for row in seqreader:
            print(row["germline_sequence"])
            protein = translate(row["germline_sequence"])
            print(protein)

    with open(args.input_mutated, "r") as f:
        seqreader = csv.DictReader(f, delimiter=',')
        for row in seqreader:
            print(row["sequence"])
            protein = translate(row["sequence"])
            print(protein)



if __name__ == "__main__":
    main(sys.argv[1:])
