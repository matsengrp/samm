import pickle
import sys
import argparse
import numpy as np
import os
import os.path
import csv
import re

from survival_model_simulator import SurvivalModelSimulatorSingleColumn
from survival_model_simulator import SurvivalModelSimulatorMultiColumn
from submotif_feature_generator import SubmotifFeatureGenerator
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
    parser.add_argument('--param-path',
        type=str,
        help='parameter file path',
        default=GERMLINE_PARAM_FILE)
    parser.add_argument('--output-true-theta',
        type=str,
        help='true theta pickle file',
        default='_output/true_theta.pkl')
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
        help='base hazard rate in cox proportional hazards model for a single motif (summing over targets)',
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
        default=1)
    parser.add_argument('--ratio-nonzero',
        type=float,
        help='Proportion of motifs that are nonzero',
        default=0.1)
    parser.add_argument('--theta-sampling-range',
        type=float,
        help='Range over which to sample theta [-r, r]',
        default=2)
    parser.add_argument('--guarantee-motifs-showup',
        action="store_true",
        help='Make sure the nonzero motifs show up in the germline')
    parser.add_argument('--per-target-model',
        action="store_true",
        help='Allow different hazard rates for different target nucleotides')
    parser.add_argument('--with-replacement',
        action="store_true",
        help='Allow same position to mutate multiple times')

    parser.set_defaults(guarantee_motifs_showup=False, per_target_model=False, with_replacement=False)
    args = parser.parse_args()
    # Only even random gene lengths allowed
    assert(args.random_gene_len % 2 == 0)
    # Only odd motif lengths allowed
    assert(args.motif_len % 2 == 1)

    if args.per_target_model:
        args.lambda0 /= 3

    return args

def _generate_true_parameters(feature_vec_len, motif_list, args):
    # True vector
    num_theta_col = NUM_NUCLEOTIDES if args.per_target_model else 1
    true_thetas = np.zeros((feature_vec_len, num_theta_col))
    probability_matrix = None

    possible_motif_mask = get_possible_motifs_to_targets(motif_list, (feature_vec_len, NUM_NUCLEOTIDES))
    impossible_motif_mask = ~possible_motif_mask

    if args.per_target_model:
        # Set the impossible thetas to -inf
        true_thetas[impossible_motif_mask] = -np.inf
    else:
        # True probabilities of mutating to a certain target nucleotide
        # Only relevant when we deal with a single theta column
        # Suppose equal probability to all target nucleotides for zero motifs
        probability_matrix = np.ones((feature_vec_len, NUM_NUCLEOTIDES))/3.0
        probability_matrix[impossible_motif_mask] = 0

    some_num_nonzero_motifs = int(true_thetas.shape[0] * args.ratio_nonzero/2.0)
    # Remove edge motifs from random choices
    # Also cannot pick the last motif
    some_nonzero_motifs = np.random.choice(true_thetas.shape[0] - 1, some_num_nonzero_motifs)
    # Also make some neighbor motifs nonzero
    nonzero_motifs = []
    for idx in some_nonzero_motifs:
        center_nucleotide_idx = NUCLEOTIDE_DICT[motif_list[idx][args.motif_len/2]]
        if not args.per_target_model:
            true_thetas[idx] = (np.random.rand() - 0.5) * args.theta_sampling_range * 2
            probability_matrix[idx,:] = np.random.rand(4)
            probability_matrix[idx, center_nucleotide_idx] = 0
            probability_matrix[idx,:] /= np.sum(probability_matrix[idx,:])

            nonzero_motifs.append(motif_list[idx])

            center_nucleotide_idx_next = NUCLEOTIDE_DICT[motif_list[idx + 1][args.motif_len/2]]
            if center_nucleotide_idx_next == center_nucleotide_idx:
                # neighboring values also have same value (may get overridden if that motif was originally
                # set to be nonzero too)
                true_thetas[idx + 1, :] = true_thetas[idx, :]
                probability_matrix[idx + 1, :] = probability_matrix[idx, :]
                nonzero_motifs.append(motif_list[idx + 1])
                assert(probability_matrix[idx + 1, center_nucleotide_idx_next] == 0)
        else:
            true_thetas[idx, :] = (np.random.rand(NUM_NUCLEOTIDES) - 0.5) * args.theta_sampling_range * 2
            # Cannot mutate motif to a target nucleotide with the same center nucleotide.
            true_thetas[idx, center_nucleotide_idx] = -np.inf

            nonzero_motifs.append(motif_list[idx])

            center_nucleotide_idx_next = NUCLEOTIDE_DICT[motif_list[idx + 1][args.motif_len/2]]
            if center_nucleotide_idx_next == center_nucleotide_idx:
                # Only have same neighboring values if center nucleotides are the same

                # neighboring values also have same value (may get overridden if that motif was originally
                # set to be nonzero too)
                true_thetas[idx + 1, :] = true_thetas[idx, :]
                nonzero_motifs.append(motif_list[idx + 1])
                assert(true_thetas[idx + 1, center_nucleotide_idx_next] == -np.inf)

    return true_thetas, probability_matrix, nonzero_motifs

def _get_germline_nucleotides(args, nonzero_motifs=[]):
    if args.random_gene_len > 0:
        germline_genes = ["FAKE-GENE-%d" % i for i in range(args.n_germlines)]
        if args.guarantee_motifs_showup:
            num_nonzero_motifs = len(nonzero_motifs)
            germline_nucleotides = [get_random_dna_seq(args.random_gene_len + args.motif_len) for i in range(args.n_germlines - num_nonzero_motifs)]
            # Let's make sure that our nonzero motifs show up in a germline sequence at least once
            for motif in nonzero_motifs:
                new_str = get_random_dna_seq(args.random_gene_len/2) + motif + get_random_dna_seq(args.random_gene_len/2)
                germline_nucleotides.append(new_str)
        else:
            # If there are very many germlines, just generate random DNA sequences
            germline_nucleotides = [get_random_dna_seq(args.random_gene_len + args.motif_len) for i in range(args.n_germlines)]
    else:
        # Read parameters from file
        params = read_germline_file(args.param_path)

        # Select, with replacement, args.n_germlines germline genes from our
        # parameter file and place them into a numpy array.
        germline_genes = np.random.choice(params.index.values, size=args.n_germlines)

        # Put the nucleotide content of each selected germline gene into a
        # corresponding list.
        germline_nucleotides = [row[gene] for gene in germline_genes]

    return germline_nucleotides, germline_genes

def dump_parameters(true_thetas, probability_matrix, args, motif_list):
    # Dump a pickle file of simulation parameters
    pickle.dump([true_thetas, probability_matrix], open(args.output_true_theta, 'w'))

    # Dump a text file of theta for easy viewing
    with open(re.sub('.pkl', '.txt', args.output_true_theta), 'w') as f:
        f.write("True Theta\n")
        lines = get_nonzero_theta_print_lines(true_thetas, motif_list)
        f.write(lines)

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

    # Randomly generate number of mutations or use default
    np.random.seed(args.seed)

    feat_generator = SubmotifFeatureGenerator(motif_len=args.motif_len)
    motif_list = feat_generator.get_motif_list()

    true_thetas, probability_matrix, nonzero_motifs = _generate_true_parameters(feat_generator.feature_vec_len, motif_list, args)

    germline_nucleotides, germline_genes = _get_germline_nucleotides(args, nonzero_motifs)

    if args.per_target_model:
        simulator = SurvivalModelSimulatorMultiColumn(true_thetas, feat_generator, lambda0=args.lambda0)
    else:
        simulator = SurvivalModelSimulatorSingleColumn(true_thetas, probability_matrix, feat_generator, lambda0=args.lambda0)

    dump_parameters(true_thetas, probability_matrix, args, motif_list)

    dump_germline_data(germline_nucleotides, germline_genes, args)

    # For each germline gene, run survival model to obtain mutated sequences.
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
                    censoring_time=args.min_censor_time + 0.1 * np.random.rand(), # allow some variation in censor time
                    with_replacement=args.with_replacement,
                ) for i in range(args.n_taxa)
            ]

            # write to file in csv format
            for i, sample in enumerate(full_data_samples):
                seq_file.writerow([gene, "%s-sample-%d" % (gene, i) , sample.left_flank + sample.end_seq + sample.right_flank])

if __name__ == "__main__":
    main(sys.argv[1:])
