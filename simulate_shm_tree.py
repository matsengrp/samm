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

from gctree.bin.mutation_model import MutationModel

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
        data file path, if --n-taxa and --n-clonal-families unspecified then
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
    parser_simulate.add_argument('--output-germline-seqs',
        type=str,
        help='germline genes used in csv file',
        default='_output/genes.csv')
    parser_simulate.add_argument('--log-dir',
        type=str,
        help='log directory',
        default='_output')
    parser_simulate.add_argument('--n-clonal-families',
        type=int,
        help='number of clonal families to generate',
        default=None)
    parser_simulate.add_argument('--verbose',
        action='store_true',
        help='output R log')
    parser_simulate.add_argument('--motif-lens',
        type=str,
        default='3,5',
        help='comma-separated motif lengths (odd only)')
    parser_simulate.add_argument('--lambda-mean',
        type=float,
        default=.01,
        help='mean of poisson to generate num mutations')
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
    parser_simulate.add_argument('--output-per-branch-germline-seqs',
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
    parser_simulate.add_argument('--mutability',
        type=str,
        default='gctree/S5F/Mutability.csv',
        help='path to mutability model file')
    parser_simulate.add_argument('--substitution',
        type=str,
        default='gctree/S5F/Substitution.csv',
        help='path to substitution model file')
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
    parser_simulate.add_argument('--locus',
        type=str,
        default='',
        help="which locus to get statistics from")
    parser_simulate.set_defaults(func=simulate, subcommand=simulate)

    ###
    # for parsing

    args = parser.parse_args()
    args.motif_lens = sorted(map(int, args.motif_lens.split(",")))

    return args

def run_gctree(args, germline_seq, mutation_model, n_taxa):
    ''' somewhat cannibalized gctree simulation '''

    mutability_sum = 0.
    with open(args.mutability, 'r') as f:
        f.next()
        for line in f:
            motif, score = line.replace('"', '').split()[:2]
            mutability_sum += float(score)

    lambda0 = [args.lambda_mean * len(germline_seq) * len(germline_seq) / mutability_sum]

    tree = mutation_model.simulate(germline_seq,
                                   lambda0=lambda0,
                                   N=n_taxa,
                                   T=args.T,
                                   progeny=poisson(.9, loc=1))

    return tree

def _get_germline_info(args):
    germline_info = []
    if args.use_partis:
        out_dir = os.path.dirname(os.path.realpath(args.output_germline_seqs))
        g = GermlineSimulatorPartis(output_dir=out_dir)
        germline_seqs, germline_freqs = g.generate_germline_set()
        germline_info = []
        for name in germline_seqs.keys():
            germline_info.append({
                'gene_name': name,
                'germline_sequence': germline_seqs[name],
                'freq': germline_freqs[name],
            })
    else:
        # Read parameters from file
        params = read_germline_file(args.param_path)

        # Find germline genes with "N" and remove them so gctree is happy
        genes_to_sample = [idx for idx, row in params.iterrows() if set(row['base'].lower()) == NUCLEOTIDE_SET]

        # Select, with replacement, args.n_clonal_families germline genes from our
        # parameter file and place them into a numpy array.
        germline_genes = np.random.choice(genes_to_sample, size=args.n_clonal_families)

        # Put the nucleotide content of each selected germline gene into a
        # corresponding list.
        for name, nucs in zip(germline_genes, [params.loc[gene]['base'] for gene in germline_genes]):
            germline_info.append({
                'gene_name': name,
                'germline_sequence': nucs,
                'freq': 1.0/args.n_clonal_families,
            })

    return germline_info

def _get_clonal_family_stats(path_to_annotations, metadata, use_np=False, use_immunized=False, locus=''):
    '''
    get data statistics from partis annotations

    @param path_to_annotations: path to partis annotations
    @param metadata: path to partis metadata 
    @param use_np: use nonproductive seqs?
    @param use_immunized: for Cui data, use immunized mice?
    @param locus: which locus to use

    @return list of clonal family sizes from processed data
    '''

    partition_info = get_partition_info(
        path_to_annotations,
        metadata,
    )

    if use_np:
        # return only nonproductive sequences
        # here "nonproductive" is defined as having a stop codon or being
        # out of frame or having a mutated conserved cysteine
        good_seq = lambda seqs: seqs['stops'] or not seqs['in_frames'] or seqs['mutated_invariants']
    else:
        # return all sequences
        good_seq = lambda seqs: [True for seq in seqs['seqs']]

    clonal_family_sizes = []
    for data_idx, data_info in enumerate(partition_info):
        if use_immunized and data_info['group'] != 'immunized':
            continue
        if not locus or data_info['locus'] != locus:
            continue
        glfo = glutils.read_glfo(data_info['germline_file'], locus=data_info['locus'])
        with open(data_info['annotations_file'][0], "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for idx, line in enumerate(reader):
                # add goodies from partis
                if len(line['input_seqs']) == 0:
                    # sometimes data will have empty clusters
                    continue
                process_input_line(line)
                add_implicit_info(glfo, line)
                good_seq_idx = [i for i, is_good in enumerate(good_seq(line)) if is_good]
                if not good_seq_idx:
                    # no nonproductive sequences... skip
                    continue
                else:
                    clonal_family_sizes.append(len(good_seq_idx))

    return clonal_family_sizes

def simulate(args):
    ''' simulate submodule '''

    # write empty sequence file before appending
    output_dir, _ = os.path.split(args.output_seqs)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Randomly generate number of mutations or use default
    np.random.seed(args.seed)

    if args.n_taxa is None:
        clonal_family_sizes = _get_clonal_family_stats(args.path_to_annotations, args.path_to_metadata, use_np=args.use_np, use_immunized=args.use_immunized, locus=args.locus)
        large_clonal_families = [n_taxa for n_taxa in clonal_family_sizes if n_taxa > args.max_taxa_per_family]
        if large_clonal_families:
            warnings.warn("There were {0} clonal families with more than {1} taxa. Ignoring: {2}".format(len(large_clonal_families), args.max_taxa_per_family, large_clonal_families))
            clonal_family_sizes = [n_taxa for n_taxa in clonal_family_sizes if n_taxa <= args.max_taxa_per_family]
        clonal_family_sizes = np.random.choice(clonal_family_sizes, args.n_clonal_families)
    else:
        clonal_family_sizes = [args.n_taxa] * args.n_clonal_families

    all_germline_dicts = _get_germline_info(args)

    mutation_model = MutationModel(args.mutability, args.substitution)

    # Write germline genes to file with two columns: name of gene and
    # corresponding sequence.
    # For each germline gene, run shmulate to obtain mutated sequences.
    # Write sequences to file with three columns: name of germline gene
    # used, name of simulated sequence and corresponding sequence.
    with open(args.output_seqs, 'w') as out_seqs, open(args.output_germline_seqs, 'w') as out_germline_seqs, \
         open(args.output_per_branch_seqs, 'w') as out_seqs_with_anc, \
         open(args.output_per_branch_germline_seqs, 'w') as out_germline_seqs_with_anc:
        gl_file = csv.writer(out_germline_seqs)
        gl_file.writerow(['germline_name','germline_sequence'])
        gl_anc_file = csv.writer(out_germline_seqs_with_anc)
        gl_anc_file.writerow(['germline_name','germline_sequence'])
        seq_file = csv.writer(out_seqs)
        seq_file.writerow(['germline_name','sequence_name','sequence'])
        seq_anc_file = csv.writer(out_seqs_with_anc)
        seq_anc_file.writerow(['germline_name','sequence_name','sequence'])
        for run in range(args.n_clonal_families):
            germline_dict = np.random.choice(all_germline_dicts, 1, p=[germline_dict['freq'] for germline_dict in all_germline_dicts])[0]
            prefix = "clone%d-" % run
            germline_name = "%s%s" % (prefix, germline_dict['gene_name'])
            # Creates a file with a single run of simulated sequences.
            # The seed is modified so we aren't generating the same
            # mutations on each run
            gl_file.writerow([germline_name, germline_dict['germline_sequence']])
            tree = run_gctree(args, germline_dict['germline_sequence'], mutation_model, clonal_family_sizes[run])
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
                    if descendant.frequency != 0 and descendant.is_leaf() and cmp(descendant.sequence.lower(), germline_dict['germline_sequence']) != 0:
                        # we are at the leaf of the tree and can write into the "observed data" file
                        obs_seq_name = "%s-%s" % (germline_name, seq_name)
                        seq_file.writerow([germline_name, obs_seq_name, descendant.sequence.lower()])

def main(args=sys.argv[1:]):
    ''' run program '''

    args = parse_args()
    args.subcommand(args)


if __name__ == "__main__":
    main(sys.argv[1:])
