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
import re


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
        default='./_output/seqs.fasta')
    parser_simulate.add_argument('--log_dir',
        type=str,
        help='log directory',
        default='./_output')
    parser_simulate.add_argument('--n_germlines',
        type=int,
        help='number of germline genes to sample',
        default=2)
    parser_simulate.add_argument('--n_mutes',
        type=int,
        help='number of mutations from germline (default: -1 meaning choose at random)',
        default=-1)

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


def run_shmulate(n_taxa, output_file, log_dir, group_name, germline, n_mutes, seed):
    ''' run shmulate through Rscript '''

    call = ['Rscript',
            'shmulate_driver.r',
            str(n_taxa),
            output_file+'_'+group_name,
            germline,
            group_name,
            str(n_mutes),
            str(seed)]

    print('Now executing:')
    print(' '.join(call))

    with open(log_dir+'/'+group_name+'.Rout', 'w') as rout:
        try:
            sout = subprocess.check_output(call, stderr=subprocess.STDOUT)
            rout.write(sout)
        except subprocess.CalledProcessError, err:
            rout.write(err.output)
            rout.write(' '.join(call))
            print(err)


def simulate(args):
    ''' simulate submodule '''

    # write empty sequence file before appending
    output_dir, _ = os.path.split(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(args.output_file, 'w') as fil:
        fil.write('')

    # Read parameters from file
    params = read_bcr_hd5(args.param_path)

    # Randomly generate number of mutations or use default
    np.random.seed(args.seed)
    if args.n_mutes < 0:
        n_mute_vec = 2 + np.random.randint(10, size=args.n_germlines)
    else:
        n_mute_vec = [args.n_mutes] * args.n_germlines

    # For each germline, run shmulate to obtain mutated sequences
    for group in range(args.n_germlines):
        gene = np.random.choice(params['gene'].unique())
        current_params = params[params['gene'] == gene]
        group_name = 'Group'+str(group+1)
        ancestor = ''.join(list(current_params['base'])).upper()

        run_shmulate(args.n_taxa, args.output_file, args.log_dir,
                group_name, ancestor, n_mute_vec[group], args.seed)

        # write to file in the BASELINe format
        with open(args.output_file+'_'+group_name, 'r') as simseqs:
            with open(args.output_file, 'a') as outseqs:
                outseqs.write('>>'+gene+'\n')
                outseqs.write(ancestor+'\n')
                for line in simseqs:
                    outseqs.write(line)


def main(args=sys.argv[1:]):
    ''' run program '''

    args = parse_args()
    args.subcommand(args)


if __name__ == "__main__":
    main(sys.argv[1:])

