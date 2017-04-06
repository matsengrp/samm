"""
Given a pickled output file with theta values, convert to csv and plot bar charts
"""

import subprocess
import sys
import argparse
import pickle
import csv

from itertools import izip
from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--input-pkl',
        type=str,
        help='pickle file with theta values')
    parser.add_argument('--output-csv',
        type=str,
        help='where to put csv output file')
    parser.add_argument('--motif-lens',
        type=str,
        help='comma-separated lengths of motifs (must all be odd)',
        default='5')
    parser.add_argument('--output-svg',
        type=str,
        help='svg file to save output to')

    args = parser.parse_args()

    return args

def convert_to_csv(target, source, motif_lens):
    """
    Take pickle file and convert to csv for use in R
    """

    with open(str(source), 'r') as f:
        theta, _ = pickle.load(f)

    feat_generator = HierarchicalMotifFeatureGenerator(motif_lens=motif_lens)
    motif_list = feat_generator.motif_list
    mutabilities = theta

    # TODO: combine hierarchical motifs here

    with open(str(target), 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(izip([motif.upper() for motif in motif_list], mutabilities.ravel()))

def main(args=sys.argv[1:]):

    args = parse_args()

    motif_len_vals = [int(m) for m in args.motif_lens.split(',')]
    for m in motif_len_vals:
        assert(m % 2 == 1)

    # TODO: maybe add a flag in case this file alread exists?
    convert_to_csv(args.output_csv, args.input_pkl, motif_len_vals)

    # Call Rscript
    command = 'Rscript'
    script_file = 'R/create_bar_plot_from_file.R'

    cmd = [command, script_file, args.output_csv, args.motif_lens, args.output_svg]
    print "Calling:", " ".join(cmd)
    res = subprocess.call(cmd)

if __name__ == "__main__":
    main(sys.argv[1:])

