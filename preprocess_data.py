import sys
import argparse
import os
import os.path
import time

from matsen_grp_data import get_paths_to_partis_annotations
from read_data import write_partis_data_from_annotations, write_data_after_imputing

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--data-path',
        type=str,
        help='location of data',
        default=None)
    parser.add_argument('--input-genes',
        type=str,
        default=None,
        help='input germline info')
    parser.add_argument('--input-seqs',
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
    parser.add_argument('--scratch-directory',
        type=str,
        help='where to write dnapars files, if necessary',
        default='_output')
    parser.add_argument('--metadata-path',
        type=str,
        help='metadata with subject/species/etc information',
        default=None)
    parser.add_argument('--output-genes',
        type=str,
        help='output germlines info')
    parser.add_argument('--output-seqs',
        type=str,
        help='output sequence info')

    args = parser.parse_args()

    assert(args.motif_len % 2 == 1 and args.motif_len > 1)

    return args

def main(args=sys.argv[1:]):
    args = parse_args()

    scratch_dir = os.path.join(args.scratch_directory, str(time.time()))
    if not os.path.exists(scratch_dir):
        os.makedirs(scratch_dir)

    if args.read_from_partis:
        write_partis_data_from_annotations(args.output_genes, args.output_seqs, args.data_path, args.metadata_path)
        if args.impute_ancestors:
            write_data_after_imputing(args.output_genes, args.output_seqs, args.output_genes, args.output_seqs, motif_len=args.motif_len, verbose=False, scratch_dir=scratch_dir)
    elif args.impute_ancestors:
        write_data_after_imputing(args.output_genes, args.output_seqs, args.input_genes, args.input_seqs, motif_len=args.motif_len, verbose=False, scratch_dir=scratch_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
