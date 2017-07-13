import sys
import argparse
import numpy as np
import csv
import subprocess

from common import *
from read_data import GERMLINE_PARAM_FILE

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed',
        type=int,
        help='rng seed for replicability',
        default=1533)
    parser.add_argument('--random-gene-len',
        type=int,
        help='Create random germline genes of this length. If zero, load true germline genes',
        default=24)
    parser.add_argument('--germline-path',
        type=str,
        help='germline file path',
        default=GERMLINE_PARAM_FILE)
    parser.add_argument('--output-file',
        type=str,
        help='simulated data destination file',
        default='_output/seqs.csv')
    parser.add_argument('--output-genes',
        type=str,
        help='germline genes used in csv file',
        default='_output/genes.csv')
    parser.add_argument('--n-germlines',
        type=int,
        help='number of germline genes to sample from (max 350)',
        default=2)
    parser.add_argument('--min-percent-mutated',
        type=float,
        help='Minimum percent of sequence to mutate',
        default=0.05)
    parser.add_argument('--max-percent-mutated',
        type=float,
        help='Maximum percent of sequence to mutate',
        default=0.20)

    parser.set_defaults(with_replacement=False)
    args = parser.parse_args()

    return args

def _get_germline_nucleotides(args, nonzero_motifs=[]):
    if args.random_gene_len > 0:
        # shmulateSeq requires codons
        args.random_gene_len -= (args.random_gene_len % 3)
        germline_genes = ["FAKE-GENE-%d" % i for i in range(args.n_germlines)]
        germline_nucleotides = [get_random_dna_seq(args.random_gene_len) for i in range(args.n_germlines)]
    else:
        # Read parameters from file
        params = read_germline_file(args.germline_path)

        # Select, with replacement, args.n_germlines germline genes from our
        # parameter file and place them into a numpy array.
        germline_genes = np.random.choice(params.index.values, size=args.n_germlines)

        # Put the nucleotide content of each selected germline gene into a
        # corresponding list.
        germline_nucleotides = [row[gene] for gene in germline_genes]

    return germline_nucleotides, germline_genes

def dump_germline_data(germline_nucleotides, germline_genes, args):
    # Write germline genes to file with two columns: name of gene and
    # corresponding sequence.
    with open(args.output_genes, 'w') as outgermlines:
        germline_file = csv.writer(outgermlines)
        germline_file.writerow(['germline_name','germline_sequence'])
        for gene, sequence in zip(germline_genes, germline_nucleotides):
            germline_file.writerow([gene,sequence])

def main(args=sys.argv[1:]):
    args = parse_args()

    germline_nucleotides, germline_genes = _get_germline_nucleotides(args)

    dump_germline_data(germline_nucleotides, germline_genes, args)

    # Call Rscript
    command = 'Rscript'
    script_file = 'R/shmulate_sequences.R'

    cmd = [command, script_file, args.output_genes, args.seed, args.min_percent_mutated, args.max_percent_mutated, args.output_file]
    print "Calling:", " ".join(map(str, cmd))
    res = subprocess.call(map(str, cmd))

if __name__ == "__main__":
    main(sys.argv[1:])
