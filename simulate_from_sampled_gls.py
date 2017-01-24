#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
    simulate sequences given germline sequences shmulate

'''

from __future__ import print_function

import subprocess
import sys
import argparse
import pandas as pd
import numpy as np
import os
import os.path
from Bio import SeqIO
import csv


def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    subparsers = parser.add_subparsers()

    ###
    # for simulating

    parser_simulate = subparsers.add_parser('simulate')

    parser_simulate.add_argument('--n_taxa',
        type=int,
        help='number of taxa to simulate',
        default=2)
    parser_simulate.add_argument('--param_path',
        type=str,
        help='parameter file path',
        default='/home/matsengrp/working/matsen/SRR1383326-annotations-imgt-v01.h5')
    parser_simulate.add_argument('--seed',
        type=int,
        help='rng seed for replicability',
        default=1533)
    parser_simulate.add_argument('--output_file',
        type=str,
        help='simulated data destination file',
        default='_output/seqs.csv')
    parser_simulate.add_argument('--output_genes',
        type=str,
        help='germline genes used in csv file',
        default='_output/genes.csv')
    parser_simulate.add_argument('--log_dir',
        type=str,
        help='log directory',
        default='_output')
    parser_simulate.add_argument('--n_germlines',
        type=int,
        help='number of germline genes to sample (maximum 350)',
        default=2)
    parser_simulate.add_argument('--n_mutes',
        type=int,
        help='number of mutations from germline (default: -1 meaning choose at random)',
        default=-1)
    parser_simulate.add_argument('--verbose',
        action='store_true',
        help='output R log')

    parser_simulate.set_defaults(subcommand=simulate)

    ###
    # for parsing

    args = parser.parse_args()

    return args


def read_bcr_hd5(path, remove_gap=True):
    ''' read hdf5 parameter file and process '''

    sites = pd.read_hdf(path, 'sites')

    if remove_gap:
        return sites.query('base != "-"')
    else:
        return sites


def run_shmulate(n_taxa, output_file, log_dir, run, germline, n_mutes, seed, verbose):
    ''' run shmulate through Rscript '''

    call = ['Rscript',
            'shmulate_driver.r',
            str(n_taxa),
            output_file+'_'+str(run),
            germline,
            'Run'+str(run),
            str(n_mutes),
            str(seed+run)]

    print('Now executing:')
    print(' '.join(call))

    try:
        sout = subprocess.check_output(call, stderr=subprocess.STDOUT)
        if verbose:
            with open(log_dir+'/'+str(run)+'.Rout', 'w') as rout:
                rout.write(sout)
    except subprocess.CalledProcessError, err:
        if verbose:
            with open(log_dir+'/'+str(run)+'.Rout', 'w') as rout:
                rout.write(err.output)
                rout.write(' '.join(call))
            print(err)

def simulate(args):
    ''' simulate submodule '''

    # write empty sequence file before appending
    output_dir, _ = os.path.split(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read parameters from file
    params = read_bcr_hd5(args.param_path)

    # Randomly generate number of mutations or use default
    np.random.seed(args.seed)

    if args.n_mutes < 0:
        n_mute_vec = 10 + np.random.randint(20, size=args.n_germlines)
    else:
        n_mute_vec = [args.n_mutes] * args.n_germlines

    # Select, with replacement, args.n_germlines germline genes from our
    # parameter file and place them into a numpy array.
    # Here 'germline_gene' is something like IGHV1-2*01.
    germline_genes = np.random.choice(params['gene'].unique(),
            size=args.n_germlines)

    # Put the nucleotide content of each selected germline gene into a
    # corresponding list.
    germline_nucleotides = [''.join(list(params[params['gene'] == gene]['base'])) \
            for gene in germline_genes]

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
        for run, (gene, sequence, n_mutes) in \
                enumerate(zip(germline_genes, germline_nucleotides, n_mute_vec)):
            # Creates a file with a single run of simulated sequences.
            # The seed is modified so we aren't generating the same
            # mutations on each run
            run_shmulate(args.n_taxa, args.output_file, args.log_dir,
                    run, sequence, n_mutes, args.seed, args.verbose)

            # write to file in csv format
            shmulated_seqs = SeqIO.parse(args.output_file+'_'+str(run), 'fasta')
            for seq in shmulated_seqs:
                seq_file.writerow([gene, str(seq.id), str(seq.seq)])

            os.remove(args.output_file+'_'+str(run))


def main(args=sys.argv[1:]):
    ''' run program '''

    args = parse_args()
    args.subcommand(args)


if __name__ == "__main__":
    main(sys.argv[1:])
