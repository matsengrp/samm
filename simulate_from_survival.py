import sys
import argparse
import numpy as np
import os
import os.path
import csv

from survival_model_simulator import SurvivalModelSimulator
from feature_generator import SubmotifFeatureGenerator
from common import *

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
        default=25)
    parser.add_argument('--output-true-theta',
        type=str,
        help='true theta file',
        default='_output/true_theta.txt')
    parser.add_argument('--output-file',
        type=str,
        help='simulated data destination file',
        default='_output/seqs.csv')
    parser.add_argument('--output-genes',
        type=str,
        help='germline genes used in csv file',
        default='_output/genes.csv')
    parser.add_argument('--lambda0',
        type=float,
        help='base hazard rate in cox proportional hazards model',
        default=0.1)
    parser.add_argument('--n-taxa',
        type=int,
        help='number of taxa to simulate',
        default=2)
    parser.add_argument('--n-germlines',
        type=int,
        help='number of germline genes to sample from (max 350)',
        default=2)
    parser.add_argument('--motif-len',
        type=int,
        help='length of motif (must be odd)',
        default=5)
    parser.add_argument('--min-censor-time',
        type=float,
        help='Minimum censoring time',
        default=0.1)
    parser.add_argument('--ratio-nonzero',
        type=float,
        help='Proportion of motifs that are nonzero',
        default=0.1)

    args = parser.parse_args()
    return args

def main(args=sys.argv[1:]):
    args = parse_args()

    # Randomly generate number of mutations or use default
    np.random.seed(args.seed)

    feat_generator = SubmotifFeatureGenerator(submotif_len=args.motif_len)
    motif_list = feat_generator.get_motif_list()
    # True vector
    true_theta = np.matrix(np.zeros(feat_generator.feature_vec_len)).T

    # Hard code simulation to have edge motifs with higher mutation rates
    true_theta[feat_generator.feature_vec_len - 1] = 0.1

    some_num_nonzero_motifs = int(true_theta.size * args.ratio_nonzero/2.0)
    # Remove edge motifs from random choices
    some_nonzero_motifs = np.random.choice(true_theta.size - 1, some_num_nonzero_motifs)
    # Also make some neighbor motifs nonzero
    nonzero_motifs = np.unique(np.vstack((some_nonzero_motifs, some_nonzero_motifs + 1)))
    num_nonzero_motifs = nonzero_motifs.size
    for idx in some_nonzero_motifs:
        # randomly set nonzero indices between [-2, 2]
        true_theta[idx] = (np.random.rand() - 0.5) * 4
        # neighboring values also have same value (may get overridden if that motif was originally
        # set to be nonzero too)
        true_theta[idx + 1] = true_theta[idx]

    if args.random_gene_len > 0:
        germline_genes = ["FAKE_GENE_%d" % i for i in range(args.n_germlines)]
        germline_nucleotides = [get_random_dna_seq(args.random_gene_len) for i in range(args.n_germlines - num_nonzero_motifs)]
        # Let's make sure that our nonzero motifs show up in a germline sequence at least once
        for motif_idx in nonzero_motifs:
            new_str = get_random_dna_seq(args.random_gene_len/2) + motif_list[motif_idx] + get_random_dna_seq(args.random_gene_len/2)
            germline_nucleotides.append(new_str)
    else:
        # Read germline genes from this file
        params = read_bcr_hd5(GERMLINE_PARAM_FILE)

        # Select, with replacement, args.n_germlines germline genes from our
        # parameter file and place them into a numpy array.
        # Here 'germline_gene' is something like IGHV1-2*01.
        germline_genes = np.random.choice(params['gene'].unique(),
                size=args.n_germlines)

        # Put the nucleotide content of each selected germline gene into a
        # corresponding list.
        germline_nucleotides = [''.join(list(params[params['gene'] == gene]['base'])) \
                for gene in germline_genes]

    simulator = SurvivalModelSimulator(true_theta, feat_generator, lambda0=args.lambda0)

    with open(args.output_true_theta, 'w') as f:
        f.write("True Theta\n")
        for i in range(true_theta.size):
            if np.abs(true_theta[i]) > ZERO_THRES:
                if i == true_theta.size - 1:
                    f.write("%d: %f (EDGES)\n" % (i, true_theta[i]))
                else:
                    f.write("%d: %f (%s)\n" % (i, true_theta[i], motif_list[i]))

    # Write germline genes to file with two columns: name of gene and
    # corresponding sequence.
    with open(args.output_genes, 'w') as outgermlines:
        germline_file = csv.writer(outgermlines)
        germline_file.writerow(['germline_name','germline_sequence'])
        for gene, sequence in zip(germline_genes, germline_nucleotides):
            germline_file.writerow([gene,sequence])

    # For each germline gene, run shmulate to obtain mutated sequences.
    # Write sequences to file with three columns: name of germline gene
    # used, name of simulated sequence and corresponding sequence.
    with open(args.output_file, 'w') as outseqs:
        seq_file = csv.writer(outseqs)
        seq_file.writerow(['germline_name','sequence_name','sequence'])
        for run, (gene, sequence) in \
                enumerate(zip(germline_genes, germline_nucleotides)):

            full_data_samples = [
                simulator.simulate(
                    start_seq=sequence.lower(),
                    censoring_time=args.min_censor_time + np.random.rand(), # censoring time at least 0.1
                ) for i in range(args.n_taxa)
            ]

            # write to file in csv format
            for i, sample in enumerate(full_data_samples):
                seq_file.writerow([gene, "%s-sample-%d" % (gene, i) , sample.obs_seq_mutation.end_seq])

if __name__ == "__main__":
    main(sys.argv[1:])
