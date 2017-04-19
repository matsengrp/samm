import pickle
import sys
import argparse
import itertools
import numpy as np
import os
import os.path
import csv
import re
import random

from survival_model_simulator import SurvivalModelSimulatorSingleColumn
from survival_model_simulator import SurvivalModelSimulatorMultiColumn
from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator
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
    parser.add_argument('--mutability',
        type=str,
        default='gctree/S5F/Mutability.csv',
        help='path to mutability model file')
    parser.add_argument('--substitution',
        type=str,
        default='gctree/S5F/Substitution.csv',
        help='path to substitution model file')
    parser.add_argument('--germline-path',
        type=str,
        help='germline file path',
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
    parser.add_argument('--sparsity-ratio',
        type=float,
        help='Proportion of motifs to set to zero',
        default=0.5)
    parser.add_argument('--guarantee-motifs-showup',
        action="store_true",
        help='Make sure the nonzero motifs show up in the germline')
    parser.add_argument('--per-target-model',
        action="store_true",
        help='Allow different hazard rates for different target nucleotides')
    parser.add_argument('--with-replacement',
        action="store_true",
        help='Allow same position to mutate multiple times')
    parser.add_argument('--shuffle',
        action="store_true",
        help='Use a shuffled version of the S5F parameters')
    parser.add_argument('--hierarchical',
        action="store_true",
        help='Generate parameters in a hierarchical manner')

    parser.set_defaults(guarantee_motifs_showup=False, per_target_model=False, with_replacement=False, shuffle=False, hierarchical=False)
    args = parser.parse_args()
    # Only even random gene lengths allowed
    assert(args.random_gene_len % 2 == 0)
    # Only odd motif lengths allowed
    assert(args.motif_len % 2 == 1)

    if args.per_target_model:
        args.lambda0 /= 3

    return args

def _read_mutability_probability_params(motif_list, args):
    """
    Read S5F parameters and use a shuffled version if requested
    Note: a shuffled version should prevent bias when comparing
    performance between shazam and our survival model since the
    current S5F parameters were estimated by counting and then
    aggregating inner 3-mers.
    """
    theta_dict = {}
    with open(args.mutability, "rb") as mutability_f:
        mut_reader = csv.reader(mutability_f, delimiter=" ")
        mut_reader.next()
        for row in mut_reader:
            motif = row[0].lower()
            motif_hazard = np.log(float(row[1]))
            theta_dict[motif] = motif_hazard

    if args.shuffle:
        random.shuffle(motif_list)
    theta = np.array([theta_dict[m] for m in motif_list])

    substitution_dict = {}
    with open(args.substitution, "rb") as substitution_f:
        substitution_reader = csv.reader(substitution_f, delimiter=" ")
        substitution_reader.next()
        for row in substitution_reader:
            motif = row[0].lower()
            substitution_dict[motif] = [float(row[i]) for i in range(1, 5)]
    probability_matrix = np.array([substitution_dict[m] for m in motif_list])

    if args.per_target_model:
        # Create the S5F target matrix and use this as our true theta
        theta = np.diag(np.exp(theta)) * np.hstack([np.ones((len(motif_list), 1)), np.matrix(probability_matrix)])
        theta_mask = get_possible_motifs_to_targets(motif_list, theta.shape)
        theta[~theta_mask] = 1 # set to 1 so i can take a log
        theta = np.log(theta)
        theta[~theta_mask] = -np.inf # set to -inf since cannot transition to this target nucleotide

    num_cols = NUM_NUCLEOTIDES + 1 if args.per_target_model else 1
    return theta.reshape((len(motif_list), num_cols)), probability_matrix

def random_generate_thetas(motif_list, motif_lens, per_target_model):
    """
    Returns back an aggregated theta vector corresponding to the motif_list as well as the raw theta vector
    """
    max_motif_len = max(motif_lens)
    num_theta_cols = NUM_NUCLEOTIDES + 1 if per_target_model else 1

    feat_generator = HierarchicalMotifFeatureGenerator(motif_lens=motif_lens)
    true_theta = np.zeros((len(motif_list), num_theta_cols))
    full_raw_theta = np.zeros((feat_generator.feature_vec_len, num_theta_cols))

    theta_scaling = 1.0
    start_idx = 0
    for f in feat_generator.feat_gens:
        motif_list = f.motif_list
        sub_theta = theta_scaling * (2 * np.random.rand(len(motif_list)) - 1)
        full_raw_theta[start_idx : start_idx + f.feature_vec_len, 0] = sub_theta
        if per_target_model:
            sub_target_theta = theta_scaling * 0.1 * np.random.randn(f.feature_vec_len, NUM_NUCLEOTIDES)
            sub_target_theta_mask = get_possible_motifs_to_targets(motif_list, sub_target_theta.shape)
            sub_target_theta[~sub_target_theta_mask] = -np.inf
            full_raw_theta[start_idx : start_idx + f.feature_vec_len, 1:] = sub_target_theta
        start_idx += f.feature_vec_len
        for i, motif in enumerate(motif_list):
            flank_len = max_motif_len - f.motif_len
            motif_flanks = itertools.product(*([NUCLEOTIDES] * flank_len))
            for flank in motif_flanks:
                full_motif = "".join(flank[:flank_len/2]) + motif + "".join(flank[-flank_len/2:])
                full_motif_idx = feat_generator.feat_gens[-1].motif_dict[full_motif]
                true_theta[full_motif_idx, 0] += sub_theta[i]
                if per_target_model:
                    true_theta[full_motif_idx, 1:] += sub_target_theta[i,:]
        theta_scaling *= 0.5

    return true_theta, full_raw_theta

def _generate_true_parameters(motif_list, args):
    """
    Read mutability and substitution parameters from S5F
    Make a sparse version if sparsity_ratio > 0
    """
    nonzero_motifs = motif_list
    if args.motif_len == 5 and not args.hierarchical:
        true_theta, probability_matrix = _read_mutability_probability_params(motif_list, args)
        raw_theta = true_theta
    else:
        if args.hierarchical:
            motif_lens = range(3, args.motif_len + 1, 2)
        else:
            motif_lens = [args.motif_len]
        true_theta, raw_theta = random_generate_thetas(motif_list, motif_lens, args.per_target_model)

        if not args.per_target_model:
            probability_matrix = np.ones((len(motif_list), NUM_NUCLEOTIDES)) * 1.0/3
            for idx, m in enumerate(motif_list):
                center_nucleotide_idx = NUCLEOTIDE_DICT[m[args.motif_len/2]]
                probability_matrix[idx, center_nucleotide_idx] = 0
        else:
            probability_matrix = None

    if args.sparsity_ratio > 0:
        # Let's zero out some motifs now
        num_zero_motifs = int(args.sparsity_ratio * len(motif_list))
        zero_motif_idxs = np.random.choice(len(motif_list), size=num_zero_motifs, replace=False)
        zero_motifs = set()
        for idx in zero_motif_idxs:
            zero_motifs.add(motif_list[idx])
            true_theta[idx,] = 0
            center_nucleotide_idx = NUCLEOTIDE_DICT[motif_list[idx][args.motif_len/2]]
            if args.per_target_model:
                true_theta[idx, center_nucleotide_idx] = -np.inf
            else:
                probability_matrix[idx, :] = 1./3
                probability_matrix[idx, center_nucleotide_idx] = 0
        nonzero_motifs = list(set(motif_list) - zero_motifs)
    return true_theta, probability_matrix, nonzero_motifs, raw_theta

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
        params = read_germline_file(args.germline_path)

        # Select, with replacement, args.n_germlines germline genes from our
        # parameter file and place them into a numpy array.
        germline_genes = np.random.choice(params.index.values, size=args.n_germlines)

        # Put the nucleotide content of each selected germline gene into a
        # corresponding list.
        germline_nucleotides = [row[gene] for gene in germline_genes]

    return germline_nucleotides, germline_genes

def dump_parameters(true_thetas, probability_matrix, raw_theta, args, motif_list):
    # Dump a pickle file of simulation parameters
    pickle.dump([true_thetas, probability_matrix, raw_theta], open(args.output_true_theta, 'w'))

    # Dump a text file of theta for easy viewing
    with open(re.sub('.pkl', '.txt', args.output_true_theta), 'w') as f:
        f.write("True Theta\n")
        lines = get_nonzero_theta_print_lines(true_thetas, motif_list, args.motif_len)
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

    feat_generator = HierarchicalMotifFeatureGenerator(motif_lens=[args.motif_len])
    motif_list = feat_generator.motif_list

    true_thetas, probability_matrix, nonzero_motifs, raw_theta = _generate_true_parameters(motif_list, args)

    germline_nucleotides, germline_genes = _get_germline_nucleotides(args, nonzero_motifs)

    if args.per_target_model:
        simulator = SurvivalModelSimulatorMultiColumn(true_thetas, feat_generator, lambda0=args.lambda0)
    else:
        simulator = SurvivalModelSimulatorSingleColumn(true_thetas, probability_matrix, feat_generator, lambda0=args.lambda0)

    dump_parameters(true_thetas, probability_matrix, raw_theta, args, motif_list)

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
                    censoring_time=args.min_censor_time + np.random.rand(), # allow some variation in censor time
                    with_replacement=args.with_replacement,
                ) for i in range(args.n_taxa)
            ]

            # write to file in csv format
            num_mutations = []
            for i, sample in enumerate(full_data_samples):
                num_mutations.append(len(sample.mutations))
                seq_file.writerow([gene, "%s-sample-%d" % (gene, i) , sample.left_flank + sample.end_seq + sample.right_flank])
            print "Number of mutations: %f (%f)" % (np.mean(num_mutations), np.sqrt(np.var(num_mutations)))

if __name__ == "__main__":
    main(sys.argv[1:])
