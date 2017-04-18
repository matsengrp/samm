#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
    simulate sequences given germline sequences shmulate

'''

import subprocess
import sys
import argparse
import pandas as pd
import numpy as np
import os
import os.path
import csv
import pickle

from common import *
from read_data import *
from Bio import SeqIO

from submotif_feature_generator import SubmotifFeatureGenerator
from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator
from simulate_from_survival import random_generate_thetas

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
        default=2)
    parser_simulate.add_argument('--param-path',
        type=str,
        help='parameter file path',
        default=GERMLINE_PARAM_FILE)
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
        default=2)
    parser_simulate.add_argument('--verbose',
        action='store_true',
        help='output R log')
    parser_simulate.add_argument('--motif-lens',
        type=str,
        default='3,5',
        help='comma-separated motif lengths (odd only)')
    parser_simulate.add_argument('--use-s5f',
        action="store_true",
        help="use s5f parameters")
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
    parser_simulate.add_argument('--output-per-branch-seqs',
        type=str,
        default='_output/gctree_per_branch_seqs.csv',
        help='additionally output genes from single branches with intermediate ancestors instead of leaves from germline')
    parser_simulate.set_defaults(func=simulate, use_s5f=False, subcommand=simulate)

    ###
    # for parsing

    args = parser.parse_args()
    args.motif_lens = sorted(map(int, args.motif_lens.split(",")))

    if args.use_s5f:
        args.mutability = 'gctree/S5F/Mutability.csv'
        args.substitution = 'gctree/S5F/Substitution.csv'
    else:
        args.out_dir = os.path.dirname(args.output_true_theta)
        args.mutability = "%s/mutability.csv" % args.out_dir
        args.substitution = "%s/substitution.csv" % args.out_dir
    print args

    return args


def run_gctree(args, germline_seq, mutation_model):
    ''' somewhat cannibalized gctree simulation '''

    if args.lambda0 is None:
        args.lambda0 = max([1, int(.01*len(germline_seq))])
    trials = 1000
    # this loop makes us resimulate if size too small, or backmutation
    for trial in range(trials):
        try:
            tree = mutation_model.simulate(germline_seq,
                                           lambda0=args.lambda0,
                                        #    r=args.r,
                                           N=args.n_taxa,
                                           T=args.T)
            collapsed_tree = CollapsedTree(tree=tree, frame=args.frame) # <-- this will fail if backmutations
            break
        except RuntimeError as e:
            if args.verbose:
                print('{}, trying again'.format(e))
        else:
            raise

    return tree

def generate_theta(motif_list, motif_lens, per_target_model, output_mutability, output_substitution, output_model):
    """
    Generates random theta vector/matrix and probability matrix.
    Then writes them all to file
    """
    max_motif_len = max(motif_lens)
    true_theta, raw_theta = random_generate_thetas(motif_list, motif_lens, per_target_model)
    if per_target_model:
        raise NotImplementedError()
    else:
        probability_matrix = np.ones((len(motif_list), NUM_NUCLEOTIDES)) * 1.0/3
        for idx, m in enumerate(motif_list):
            center_nucleotide_idx = NUCLEOTIDE_DICT[m[max_motif_len/2]]
            probability_matrix[idx, center_nucleotide_idx] = 0

    with open(output_model, "w") as f:
        pickle.dump((true_theta, probability_matrix, raw_theta), f)

    with open(output_mutability, "w") as f:
        csv_writer = csv.writer(f, delimiter=' ')
        csv_writer.writerow(["Motif", "Mutability"])
        for i, motif in enumerate(motif_list):
            csv_writer.writerow([motif.upper(), np.asscalar(np.exp(true_theta[i]))])

    with open(output_substitution, "w") as f:
        csv_writer = csv.writer(f, delimiter=' ')
        csv_writer.writerow(["Motif", "A", "C", "G", "T"])
        for i, motif in enumerate(motif_list):
            csv_writer.writerow([motif.upper()] + probability_matrix[i,:].tolist())

    return true_theta, probability_matrix, raw_theta

def simulate(args):
    ''' simulate submodule '''
    feat_generator = HierarchicalMotifFeatureGenerator(motif_lens=args.motif_lens)

    if not args.use_s5f:
        generate_theta(
            feat_generator.feat_gens[-1].motif_list,
            args.motif_lens,
            False,
            args.mutability,
            args.substitution,
            args.output_true_theta,
        )
        mutation_model = MutationModel(args.mutability, args.substitution)
    else:
        # using s5f
        mutation_model = MutationModel(args.mutability, args.substitution)
        context_model = mutation_model.context_model
        true_thetas = np.empty((feat_generator.feature_vec_len, 1))
        probability_matrix = np.empty((feat_generator.feature_vec_len, NUM_NUCLEOTIDES))
        for motif_idx, motif in enumerate(feat_generator.motif_list):
            mutability = context_model[motif.upper()][0]
            true_thetas[motif_idx] = np.log(mutability)
            for nuc in NUCLEOTIDES:
                probability_matrix[motif_idx, NUCLEOTIDE_DICT[nuc]] = context_model[motif.upper()][1][nuc.upper()]

        with open(args.output_true_theta, 'w') as output_theta:
            pickle.dump([true_thetas, probability_matrix], output_theta)

    # write empty sequence file before appending
    output_dir, _ = os.path.split(args.output_seqs)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read parameters from file
    params = read_germline_file(args.param_path)

    # Find genes with "N" and remove them so gctree is happy
    genes_to_sample = [idx for idx, row in params.iterrows() if set(row['base'].lower()) == NUCLEOTIDE_SET]

    # Randomly generate number of mutations or use default
    np.random.seed(args.seed)

    # Select, with replacement, args.n_germlines germline genes from our
    # parameter file and place them into a numpy array.
    germline_genes = np.random.choice(genes_to_sample, size=args.n_germlines)

    # Put the nucleotide content of each selected germline gene into a
    # corresponding list.
    germline_nucleotides = [params.loc[gene]['base'] for gene in germline_genes]

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
        for run, (gene_name, germline_sequence) in \
                enumerate(zip(germline_genes, germline_nucleotides)):
            print "run ================ %d" % run
            prefix = "clone%d-" % run
            germline_name = "%s%s" % (prefix, gene_name)
            # Creates a file with a single run of simulated sequences.
            # The seed is modified so we aren't generating the same
            # mutations on each run
            print "germline_name", germline_name
            gl_file.writerow([germline_name, germline_sequence])
            tree = run_gctree(args, germline_sequence, mutation_model)
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
                    if descendant.frequency != 0 and descendant.is_leaf() and cmp(descendant.sequence.lower(), germline_sequence) != 0:
                        # we are at the leaf of the tree and can write into the "observed data" file
                        obs_seq_name = "%s-%s" % (germline_name, seq_name)
                        seq_file.writerow([germline_name, obs_seq_name, descendant.sequence.lower()])

def main(args=sys.argv[1:]):
    ''' run program '''

    args = parse_args()
    args.subcommand(args)


if __name__ == "__main__":
    main(sys.argv[1:])
