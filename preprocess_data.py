import sys
import argparse
import os
import os.path
import time
import pandas as pd
import csv

from read_data import write_partis_data_from_annotations, write_data_after_imputing, write_data_after_sampling
from common import split_train_val
from shutil import copyfile

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--data-path',
        type=str,
        help='location of data')
    parser.add_argument('--input-genes',
        type=str,
        default=None,
        help='input germline info')
    parser.add_argument('--input-seqs',
        type=str,
        default=None,
        help='input sequence info')
    parser.add_argument('--output-genes',
        type=str,
        default=None,
        help='input germline info')
    parser.add_argument('--output-seqs',
        type=str,
        default=None,
        help='input sequence info')
    parser.add_argument('--output-train-seqs',
        type=str,
        default=None,
        help='input sequence info')
    parser.add_argument('--output-test-seqs',
        type=str,
        default=None,
        help='input sequence info')
    parser.add_argument('--motif-len',
        type=int,
        help='length of motif (must be odd)',
        default=5)
    parser.add_argument('--read-from-partis',
        action='store_true',
        help='read data from partis annotations')
    parser.add_argument('--impute-ancestors',
        action='store_true',
        help='impute ancestors using dnapars')
    parser.add_argument('--sample-from-family',
        action='store_true',
        help='sample sequence from clonal family')
    parser.add_argument("--locus",
        type=str,
        choices=('','igh','igk','igl'),
        help="locus (igh, igk or igl; default empty)",
        default='')
    parser.add_argument("--species",
        type=str,
        choices=('','mouse','human'),
        help="species (mouse or human; default empty)",
        default='')
    parser.add_argument('--scratch-directory',
        type=str,
        help='where to write dnapars files, if necessary',
        default='_output')
    parser.add_argument('--metadata-path',
        type=str,
        help='metadata with subject/species/etc information',
        default=None)
    parser.add_argument('--use-v',
        action='store_true',
        help='use V genes?')
    parser.add_argument('--use-np',
        action='store_true',
        help='use nonproductive seqs?')
    parser.add_argument('--use-immunized',
        action='store_true',
        help='use immunized mouse?')
    parser.add_argument('--test-column',
        type=str,
        help='column in the dataset to split training/testing on (e.g., subject, clonal_family, etc.)',
        default=None)
    parser.add_argument('--test-idx',
        type=int,
        help='index of test column to use for splitting (default chooses randomly based on tuning sample ratio)',
        default=None)
    parser.add_argument('--tuning-sample-ratio',
        type=float,
        help="""
            proportion of data to use for tuning the penalty parameter.
            if zero, tunes by number of confidence intervals for theta that do not contain zero
            """,
        default=0.1)

    args = parser.parse_args()

    assert(args.motif_len % 2 == 1 and args.motif_len > 1)

    return args

def write_train_test(output_seqs, sampled_set):
    """
    Write data after sampling so shazam and samm fit to the same data
    """
    with open(output_seqs, 'w') as seq_file:
        seq_writer = csv.DictWriter(seq_file, sampled_set[0].keys())
        seq_writer.writeheader()
        for seq_dict in sampled_set:
            seq_writer.writerow(seq_dict)

def main(args=sys.argv[1:]):
    args = parse_args()

    scratch_dir = os.path.join(args.scratch_directory, str(time.time()))
    if not os.path.exists(scratch_dir):
        os.makedirs(scratch_dir)

    if args.read_from_partis:
        write_partis_data_from_annotations(args.output_genes, args.output_seqs, args.data_path, args.metadata_path, use_v=args.use_v, use_np=args.use_np, use_immunized=args.use_immunized, locus=args.locus, species=args.species)
        args.input_genes = args.output_genes
        args.input_seqs = args.output_seqs

    if args.sample_from_family:
        write_data_after_sampling(args.output_genes, args.output_seqs, args.input_genes, args.input_seqs)
    elif args.impute_ancestors:
        write_data_after_imputing(args.output_genes, args.output_seqs, args.input_genes, args.input_seqs, motif_len=args.motif_len, verbose=False, scratch_dir=scratch_dir)
    elif not args.read_from_partis:
        copyfile(args.input_genes, args.output_genes)
        copyfile(args.input_seqs, args.output_seqs)

    if args.output_train_seqs is not None:
        # convert pandas df to list of dicts
        metadata = pd.read_csv(args.output_seqs).T.to_dict().values()

        train_idx, test_idx = split_train_val(
            len(metadata),
            metadata,
            args.tuning_sample_ratio,
            args.test_column,
            args.test_idx,
        )
        train_set = [metadata[i] for i in train_idx]
        test_set = [metadata[i] for i in test_idx]

        # for fitting shazam and later validating
        write_train_test(args.output_train_seqs, train_set)
        write_train_test(args.output_test_seqs, test_set)

if __name__ == "__main__":
    main(sys.argv[1:])
