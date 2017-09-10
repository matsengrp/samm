"""
Generate mutated sequences using the survival model, shmulate, or a tree given
mutation model parameters. This can generate naive sequences using partis or
construct random nucleotide sequences.
"""

import pickle
import sys
import argparse
import numpy as np
import os
import os.path
import csv
import subprocess

from survival_model_simulator import SurvivalModelSimulatorSingleColumn
from survival_model_simulator import SurvivalModelSimulatorMultiColumn
from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator
from simulate_germline import GermlineSimulatorPartis, GermlineMetadata
from common import *

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed',
        type=int,
        help='Random number generator seed for replicability',
        default=1)
    parser.add_argument('--random-gene-len',
        type=int,
        help='Create random naive sequences of this length, only used if not using partis',
        default=24)
    parser.add_argument('--agg-motif-len',
        type=int,
        help='Length of k-mer motifs in the aggregate model -- assumes that the center position mutates',
        default=3)
    parser.add_argument('--input-model',
        type=str,
        help='Input file with true theta parameters',
        default='_output/true_model.pkl')
    parser.add_argument('--use-partis',
        action="store_true",
        help='Use partis to gernerate germline/naive sequences')
    parser.add_argument('--use-shmulate',
        action="store_true",
        help='Use shmulate to do somatic hypermutation simulation')
    parser.add_argument('--output-mutated',
        type=str,
        help='CSV file to output for mutated sequences',
        default='_output/mutated.csv')
    parser.add_argument('--output-naive',
        type=str,
        help='CSV file to output for naive sequences',
        default='_output/naive.csv')
    parser.add_argument('--lambda0',
        type=float,
        help='Baseline constant hazard rate in cox proportional hazards model',
        default=0.1)
    parser.add_argument('--tot-mutated',
        type=int,
        help='Roughly total number of mutated sequences',
        default=1)
    parser.add_argument('--n-naive',
        type=int,
        help='Number of naive sequences to create, only used if not using partis',
        default=2)
    parser.add_argument('--n-subjects',
        type=int,
        help='Number of subjects (so number of germline sets) - used by partis',
        default=1)
    parser.add_argument('--min-percent-mutated',
        type=float,
        help='Minimum percent of sequence to mutate',
        default=0.01)
    parser.add_argument('--max-percent-mutated',
        type=float,
        help='Maximum percent of sequence to mutate',
        default=0.05)
    parser.add_argument('--with-replacement',
        action="store_true",
        help='Allow same position to mutate multiple times')

    parser.set_defaults(with_replacement=False, use_partis=False, use_shmulate=False)
    args = parser.parse_args()
    args.output_naive_freqs = args.output_naive.replace(".csv", "_prevalence.csv")
    return args

def _get_germline_nucleotides(args, nonzero_motifs=[]):
    if args.use_partis:
        out_dir = os.path.dirname(os.path.realpath(args.output_naive))
        g = GermlineSimulatorPartis(output_dir=out_dir)
        germline_seqs = g.generate_germline_sets(num_sets=args.n_subjects)
    else:
        # generate germline sequences at random by drawing from ACGT multinomial
        # suppose all alleles have equal frequencies
        germline_genes = ["FAKE-GENE-%d" % i for i in range(args.n_naive)]
        germline_nucleotides = [get_random_dna_seq(args.random_gene_len) for i in range(args.n_naive)]
        germline_seqs = {g:GermlineMetadata(n, g, 1.0/args.n_naive, g) for g,n in zip(germline_genes, germline_nucleotides)}

    return germline_seqs

def dump_germline_data(germline_seqs, args):
    # Write germline genes to file with two columns: name of gene and
    # corresponding sequence.
    with open(args.output_naive, 'w') as outgermlines:
        germline_file = csv.writer(outgermlines)
        germline_file.writerow(['germline_name','germline_sequence', 'germline_family'])
        for gene, seq_metadata in germline_seqs.iteritems():
            germline_file.writerow([gene, seq_metadata.val, seq_metadata.family])

    with open(args.output_naive_freqs, 'w') as outgermlines:
        germline_freq_file = csv.writer(outgermlines)
        germline_freq_file.writerow(['germline_name','freq'])
        for gene, seq_metadata in germline_seqs.iteritems():
            germline_freq_file.writerow([gene, seq_metadata.freq])

def run_survival(args, germline_seqs):
    feat_generator = HierarchicalMotifFeatureGenerator(
        motif_lens=[args.agg_motif_len],
        left_motif_flank_len_list=[[args.agg_motif_len/2]],
    )
    with open(args.input_model, 'r') as f:
        agg_theta, _ = pickle.load(f)

    if agg_theta.shape[1] == NUM_NUCLEOTIDES:
        simulator = SurvivalModelSimulatorMultiColumn(agg_theta, feat_generator, lambda0=args.lambda0)
    elif agg_theta.shape[1] == 1:
        agg_theta_shape = (agg_theta.size, NUM_NUCLEOTIDES)
        probability_matrix = np.ones(agg_theta_shape) * 1.0/3
        possible_motifs_mask = get_possible_motifs_to_targets(feat_generator.motif_list, agg_theta_shape, feat_generator.mutating_pos_list)
        probability_matrix[~possible_motifs_mask] = 0
        simulator = SurvivalModelSimulatorSingleColumn(agg_theta, probability_matrix, feat_generator, lambda0=args.lambda0)
    else:
        raise ValueError("Aggregate theta shape is wrong")

    # For each germline gene, run survival model to obtain mutated sequences.
    # Write sequences to file with three columns: name of germline gene
    # used, name of simulated sequence and corresponding sequence.
    tot_mutated = 0
    tot_germline = len(germline_seqs)
    with open(args.output_mutated, 'w') as outseqs:
        seq_file = csv.writer(outseqs)
        seq_file.writerow(['germline_name', 'sequence_name', 'sequence'])
        germline_keys = germline_seqs.keys()
        mult_sample = np.random.multinomial(
            args.tot_mutated,
            [germline_seqs[g_key].freq for g_key in germline_keys],
        )
        for idx, gene in enumerate(germline_keys):
            sequence = germline_seqs[gene].val
            # Decide amount to mutate -- just random uniform
            percent_to_mutate = np.random.uniform(low=args.min_percent_mutated, high=args.max_percent_mutated)
            # Decide number of taxa. Must be at least one.
            n_germ_taxa = mult_sample[idx]
            if n_germ_taxa > 0:
                full_data_samples = [
                    simulator.simulate(
                        start_seq=sequence.lower(),
                        percent_mutated=percent_to_mutate,
                        with_replacement=args.with_replacement,
                    ) for i in range(n_germ_taxa)
                ]

                # write to file in csv format
                num_mutations = []
                for i, sample in enumerate(full_data_samples):
                    num_mutations.append(len(sample.mutations))
                    seq_file.writerow([gene, "%s-sample-%d" % (gene, i) , sample.left_flank + sample.end_seq + sample.right_flank])
                print "Number of mutations: %f (%f)" % (np.mean(num_mutations), np.sqrt(np.var(num_mutations)))

def run_shmulate(args):
    # Call Rscript
    command = 'Rscript'
    script_file = 'R/shmulate_sequences.R'

    cmd = [
        command,
        script_file,
        args.output_naive,
        args.output_naive_freqs,
        args.tot_mutated,
        args.seed,
        args.min_percent_mutated,
        args.max_percent_mutated,
        args.output_mutated,
    ]
    print "Calling:", " ".join(map(str, cmd))
    res = subprocess.call(map(str, cmd))
    print """
        WARNING: these simulations ignore the input model.
        It uses MK_RS5NF as the targeting model.
    """

def main(args=sys.argv[1:]):
    args = parse_args()
    np.random.seed(args.seed)

    germline_seqs = _get_germline_nucleotides(args)
    dump_germline_data(germline_seqs, args)

    # If there were an even distribution, we would have this many taxa
    # But there is an uneven distribution of allele frequencies, so we will make the number of taxa
    # for different alleles to be different. The number of taxa will just be proportional to the germline
    # frequency.
    if args.use_shmulate:
        run_shmulate(args)
    else:
        run_survival(args, germline_seqs)

if __name__ == "__main__":
    main(sys.argv[1:])
