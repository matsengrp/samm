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
from simulate_germline import GermlineSimulatorPartis
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
    parser.add_argument('--agg-motif-len',
        type=int,
        help='length of motif -- assume center mutates',
        default=5)
    parser.add_argument('--input-model',
        type=str,
        default='_output/true_model.pkl',
        help='true model pickle file')
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
    parser.add_argument('--lambda0',
        type=float,
        help='base hazard rate in cox proportional hazards model for a single motif (summing over targets)',
        default=0.1)
    parser.add_argument('--n-taxa',
        type=int,
        help='average number of taxa per germline sequence',
        default=4)
    parser.add_argument('--n-germlines',
        type=int,
        help='number of germline genes to sample from (max 350). ignored if using partis',
        default=2)
    parser.add_argument('--min-percent-mutated',
        type=float,
        help='Minimum percent of sequence to mutate',
        default=0.05)
    parser.add_argument('--max-percent-mutated',
        type=float,
        help='Maximum percent of sequence to mutate',
        default=0.15)
    parser.add_argument('--with-replacement',
        action="store_true",
        help='Allow same position to mutate multiple times')
    parser.add_argument('--use-partis',
        action="store_true",
        help='Use partis to gernerate germline sequences')
    parser.add_argument('--use-shmulate',
        action="store_true",
        help='Use shmulate to do SHM')

    parser.set_defaults(with_replacement=False, use_partis=False, use_shmulate=False)
    args = parser.parse_args()
    args.output_gene_freqs = args.output_genes.replace(".csv", "_prevalence.csv")
    return args

def _get_germline_nucleotides(args, nonzero_motifs=[]):
    if args.use_partis:
        current_dir = os.path.dirname(os.path.realpath(__file__))
        g = GermlineSimulatorPartis(output_dir=current_dir + "/_output")
        germline_seqs, germline_freqs = g.generate_germline_set()
    else:
        # generate germline sequences at random by drawing from ACGT multinomial
        # suppose all alleles have equal frequencies
        germline_genes = ["FAKE-GENE-%d" % i for i in range(args.n_germlines)]
        germline_nucleotides = [get_random_dna_seq(args.random_gene_len) for i in range(args.n_germlines)]
        germline_seqs = {g:n for g,n in zip(germline_genes, germline_nucleotides)}
        germline_freqs = {g:1.0/args.n_germlines for g in germline_genes}

    return germline_seqs, germline_freqs

def dump_germline_data(germline_seqs, germline_freqs, args):
    # Write germline genes to file with two columns: name of gene and
    # corresponding sequence.
    with open(args.output_genes, 'w') as outgermlines:
        germline_file = csv.writer(outgermlines)
        germline_file.writerow(['germline_name','germline_sequence'])
        for gene, sequence in germline_seqs.iteritems():
            germline_file.writerow([gene,sequence])

    with open(args.output_gene_freqs, 'w') as outgermlines:
        germline_freq_file = csv.writer(outgermlines)
        germline_freq_file.writerow(['germline_name','freq'])
        for gene, freq in germline_freqs.iteritems():
            germline_freq_file.writerow([gene,freq])

def run_survival(args, germline_seqs, germline_freqs):
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
    with open(args.output_file, 'w') as outseqs:
        seq_file = csv.writer(outseqs)
        seq_file.writerow(['germline_name','sequence_name','sequence'])
        for gene, sequence in germline_seqs.iteritems():
            # Decide amount to mutate -- just random uniform
            percent_to_mutate = np.random.uniform(low=args.min_percent_mutated, high=args.max_percent_mutated)
            # Decide number of taxa. Must be at least one.
            n_germ_taxa = int(args.tot_taxa * germline_freqs[gene] + 1)
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

def run_shmulate(args, germline_seqs, germline_freqs):
    # Call Rscript
    command = 'Rscript'
    script_file = 'R/shmulate_sequences.R'

    cmd = [
        command,
        script_file,
        args.output_genes,
        args.output_gene_freqs,
        args.tot_taxa,
        args.seed,
        args.min_percent_mutated,
        args.max_percent_mutated,
        args.output_file,
    ]
    print "Calling:", " ".join(map(str, cmd))
    res = subprocess.call(map(str, cmd))

def main(args=sys.argv[1:]):
    args = parse_args()
    np.random.seed(args.seed)

    germline_seqs, germline_freqs = _get_germline_nucleotides(args)
    dump_germline_data(germline_seqs, germline_freqs, args)

    # If there were an even distribution, we would have this many taxa
    # But there is an uneven distribution of allele frequencies, so we will make the number of taxa
    # for different alleles to be different. The number of taxa will just be proportional to the germline
    # frequency.
    args.tot_taxa = args.n_taxa * len(germline_seqs)

    if args.use_shmulate:
        run_shmulate(args, germline_seqs, germline_freqs)
    else:
        run_survival(args, germline_seqs, germline_freqs)

if __name__ == "__main__":
    main(sys.argv[1:])
