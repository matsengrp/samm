#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
    simulate sequences given germline sequences shmulate

'''

import subprocess
import sys
import argparse
import numpy as np
import os
import os.path
import csv
import pickle
import warnings

from common import *
from read_data import *
from Bio import SeqIO
from scipy.stats import poisson

from submotif_feature_generator import SubmotifFeatureGenerator
from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator
from simulate_germline import GermlineSimulatorPartis

from gctree.bin.gctree import MutationModel, CollapsedTree

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    subparsers = parser.add_subparsers()

    ###
    # for simulating

    parser_simulate = subparsers.add_parser('simulate')

    parser_simulate.add_argument('--n-taxa',
        type=int,
        help='number of taxa to simulate',
        default=None)
    parser_simulate.add_argument('--param-path',
        type=str,
        help='parameter file path',
        default=GERMLINE_PARAM_FILE)
    parser_simulate.add_argument('--path-to-annotations',
        type=str,
        help='''
        data file path, if --n-taxa and --n-germlines unspecified then
        compute these statistics from supplied dataset
        ''',
        default=None)
    parser_simulate.add_argument('--path-to-metadata',
        type=str,
        help='metadata file path, same as for annotations',
        default=None)
    parser_simulate.add_argument('--seed',
        type=int,
        help='rng seed for replicability',
        default=1533)
    parser_simulate.add_argument('--output-seqs',
        type=str,
        help='simulated data destination file',
        default='_output/seqs.csv')
    parser_simulate.add_argument('--output-genes',
        type=str,
        help='germline genes used in csv file',
        default='_output/genes.csv')
    parser_simulate.add_argument('--output-true-theta',
        type=str,
        help='true theta pickle file',
        default='_output/true_theta.pkl')
    parser_simulate.add_argument('--log-dir',
        type=str,
        help='log directory',
        default='_output')
    parser_simulate.add_argument('--n-germlines',
        type=int,
        help='number of germline genes to sample (maximum 350)',
        default=None)
    parser_simulate.add_argument('--verbose',
        action='store_true',
        help='output R log')
    parser_simulate.add_argument('--motif-lens',
        type=str,
        default='3,5',
        help='comma-separated motif lengths (odd only)')
    parser_simulate.add_argument('--lambda0',
        type=float,
        default=None,
        help='baseline mutation rate')
    parser_simulate.add_argument('--r',
        type=float,
        default=1.,
        help='sampling probability')
    parser_simulate.add_argument('--T',
        type=int,
        default=None,
        help='observation time, if None we run until termination and take all leaves')
    parser_simulate.add_argument('--frame',
        type=int,
        default=None,
        help='codon frame')
    parser_simulate.add_argument('--output-per-branch-genes',
        type=str,
        default='_output/gctree_per_branch_genes.csv',
        help='additionally output genes from single branches with intermediate ancestors instead of leaves from germline')
    parser_simulate.add_argument('--max-taxa-per-family',
        type=int,
        default=1000,
        help='the maximum taxa per family to simulate when getting clonal family size statistics')
    parser_simulate.add_argument('--output-per-branch-seqs',
        type=str,
        default='_output/gctree_per_branch_seqs.csv',
        help='additionally output genes from single branches with intermediate ancestors instead of leaves from germline')
    parser_simulate.add_argument('--use-v',
        action="store_true",
        help="use V gene only")
    parser_simulate.add_argument('--use-np',
        action="store_true",
        help="use nonproductive sequences")
    parser_simulate.add_argument('--use-immunized',
        action="store_true",
        help="use immunized mice")
    parser_simulate.add_argument('--use-partis',
        action="store_true",
        help="use partis germline generation")
    parser_simulate.set_defaults(func=simulate, subcommand=simulate)

    ###
    # for parsing

    args = parser.parse_args()
    args.motif_lens = sorted(map(int, args.motif_lens.split(",")))

    args.mutability = 'gctree/S5F/Mutability.csv'
    args.substitution = 'gctree/S5F/Substitution.csv'

    return args

def run_gctree(args, germline_seq, mutation_model, n_taxa):
    ''' somewhat cannibalized gctree simulation '''

    if args.lambda0 is None:
        args.lambda0 = max([1, int(.01*len(germline_seq))])
    tree = mutation_model.simulate(germline_seq,
                                   lambda0=args.lambda0,
                                   N=n_taxa,
                                   T=args.T,
                                   progeny=poisson(.9, loc=1))

    return tree

def _get_germline_info(args):
    germline_info = []
    if args.use_partis:
        current_dir = os.path.dirname(os.path.realpath(__file__))
        g = GermlineSimulatorPartis(output_dir=current_dir + "/_output")
        germline_seqs, germline_freqs = g.generate_germline_set()
        for name in germline_seqs.keys():
            germline_info.append({
                'gene_name': name,
                'germline_sequence': germline_seqs[name],
                'freq': germline_freqs[name],
            })
    else:
        # Read parameters from file
        params = read_germline_file(args.param_path)

        # Find genes with "N" and remove them so gctree is happy
        genes_to_sample = [idx for idx, row in params.iterrows() if set(row['base'].lower()) == NUCLEOTIDE_SET]

        # Select, with replacement, args.n_germlines germline genes from our
        # parameter file and place them into a numpy array.
        germline_genes = np.random.choice(genes_to_sample, size=args.n_germlines)

        # Put the nucleotide content of each selected germline gene into a
        # corresponding list.
        for name, nucs in zip(germline_genes, [params.loc[gene]['base'] for gene in germline_genes]):
            germline_info.append({
                'gene_name': name,
                'germline_sequence': nucs,
                'freq': 1.0/args.n_germlines,
            })

    return germline_info

def simulate(args):
    ''' simulate submodule '''
    feat_generator = HierarchicalMotifFeatureGenerator(motif_lens=args.motif_lens)

    mutation_model = MutationModel(args.mutability, args.substitution)
    context_model = mutation_model.context_model
    true_thetas = np.empty((feat_generator.feature_vec_len, 1))
    probability_matrix = np.empty((feat_generator.feature_vec_len, NUM_NUCLEOTIDES))
    for motif_idx, motif in enumerate(feat_generator.motif_list):
        mutability = context_model[motif.upper()][0]
        true_thetas[motif_idx] = np.log(mutability)
        for nuc in NUCLEOTIDES:
            probability_matrix[motif_idx, NUCLEOTIDE_DICT[nuc]] = context_model[motif.upper()][1][nuc.upper()]

    probability_matrix[probability_matrix == 0] = -np.inf
    probability_matrix[probability_matrix != -np.inf] = np.log(probability_matrix[probability_matrix != -np.inf])

    with open(args.output_true_theta, 'w') as output_theta:
        pickle.dump([true_thetas, probability_matrix], output_theta)

    # write empty sequence file before appending
    output_dir, _ = os.path.split(args.output_seqs)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Randomly generate number of mutations or use default
    np.random.seed(args.seed)

    if args.n_taxa is None:
        clonal_family_sizes = get_data_stats_from_partis(args.path_to_annotations, args.path_to_metadata, use_v=args.use_v, use_np=args.use_np, use_immunized=args.use_immunized)
        large_list = [n_taxa for n_taxa in clonal_family_sizes if n_taxa > args.max_taxa_per_family]
        if large_list:
            warnings.warn("There were {0} clonal families with more than {1} taxa. Ignoring: {2}".format(len(large_list), args.max_taxa_per_family, large_list))
            clonal_family_sizes = [n_taxa for n_taxa in clonal_family_sizes if n_taxa <= args.max_taxa_per_family]

    if args.n_germlines is None:
        args.n_germlines = len(clonal_family_sizes)
    else:
        clonal_family_sizes = np.random.choice(clonal_family_sizes, args.n_germlines)

    list_of_gene_dicts = _get_germline_info(args)
    for idx, gene_dict in enumerate(list_of_gene_dicts):
        gene_dict['n_taxa'] = clonal_family_sizes[idx]

    # Write germline genes to file with two columns: name of gene and
    # corresponding sequence.
    # For each germline gene, run shmulate to obtain mutated sequences.
    # Write sequences to file with three columns: name of germline gene
    # used, name of simulated sequence and corresponding sequence.
    with open(args.output_seqs, 'w') as outseqs, open(args.output_genes, 'w') as outgermlines, \
         open(args.output_per_branch_seqs, 'w') as outseqswithanc, \
         open(args.output_per_branch_genes, 'w') as outgermlineswithanc:
        gl_file = csv.writer(outgermlines)
        gl_file.writerow(['germline_name','germline_sequence'])
        gl_anc_file = csv.writer(outgermlineswithanc)
        gl_anc_file.writerow(['germline_name','germline_sequence'])
        seq_file = csv.writer(outseqs)
        seq_file.writerow(['germline_name','sequence_name','sequence'])
        seq_anc_file = csv.writer(outseqswithanc)
        seq_anc_file.writerow(['germline_name','sequence_name','sequence'])
        for run, (gene_dict) in enumerate(list_of_gene_dicts):
            prefix = "clone%d-" % run
            germline_name = "%s%s" % (prefix, gene_dict['gene_name'])
            # Creates a file with a single run of simulated sequences.
            # The seed is modified so we aren't generating the same
            # mutations on each run
            gl_file.writerow([germline_name, gene_dict['germline_sequence']])
            tree = run_gctree(args, gene_dict['germline_sequence'], mutation_model, gene_dict['n_taxa'])
            for idx, descendant in enumerate(tree.traverse('preorder')):
                # internal nodes will have frequency zero, so for providing output
                # along a branch we need to consider these cases! otherwise the leaves
                # should have nonzero frequency
                seq_name = 'seq%d' % idx
                if descendant.is_root():
                    # Add a name to this node
                    descendant.name = germline_name
                    gl_anc_file.writerow([descendant.name, descendant.sequence.lower()])
                else:
                    # Add a name to this node
                    descendant.name = '-'.join([descendant.up.name, seq_name])
                    # Write the internal node to the tree branch germline file
                    # Note: this will write repeats, but that's okay.
                    gl_anc_file.writerow([descendant.up.name,descendant.up.sequence.lower()])
                    if cmp(descendant.sequence.lower(), descendant.up.sequence.lower()) != 0:
                        # write into the true tree branches file
                        seq_anc_file.writerow([descendant.up.name, descendant.name, descendant.sequence.lower()])
                    if descendant.frequency != 0 and descendant.is_leaf() and cmp(descendant.sequence.lower(), gene_dict['germline_sequence']) != 0:
                        # we are at the leaf of the tree and can write into the "observed data" file
                        obs_seq_name = "%s-%s" % (germline_name, seq_name)
                        seq_file.writerow([germline_name, obs_seq_name, descendant.sequence.lower()])

def main(args=sys.argv[1:]):
    ''' run program '''

    args = parse_args()
    args.subcommand(args)


if __name__ == "__main__":
    main(sys.argv[1:])
