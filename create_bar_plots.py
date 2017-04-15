"""
Given a pickled output file with theta values, convert to csv and plot bar charts
"""
import numpy as np
import subprocess
import sys
import argparse
import pickle
import csv
import itertools

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
        help='where to put csv output file',
        default='_output/out.csv')
    parser.add_argument('--motif-lens',
        type=str,
        help='comma-separated lengths of motifs (must all be odd)',
        default='3,5,7')
    parser.add_argument('--mutating-positions',
        type=str,
        help='which position in the motif is mutating; can be one of combination of -1, 0, 1 for 5\'/left end, central, or 3\'/right end',
        default='-1,0,1')
    parser.add_argument('--output-svg',
        type=str,
        help='svg file to save output to',
        default='_output/out.png')

    args = parser.parse_args()

    return args

def convert_to_csv(target, mutabilities, motif_lens, mutating_positions):
    """
    Take pickle file and convert to csv for use in R
    """
    feat_generator = HierarchicalMotifFeatureGenerator(
            motif_lens=motif_lens,
            mutating_positions=mutating_positions,
        )
    motif_list = feat_generator.motif_list
    mutating_pos_list = feat_generator.mutating_pos_list

    with open(str(target), 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(izip([motif.upper() for motif in motif_list], mutating_pos_list, mutabilities.ravel()))

def main(args=sys.argv[1:]):

    args = parse_args()

    motif_len_vals = [int(m) for m in args.motif_lens.split(',')]
    for m in motif_len_vals:
        assert(m % 2 == 1)

    mutating_pos_vals = [int(pos) for pos in args.mutating_positions.split(',')]
    for m in mutating_pos_vals:
        assert(m in [-1, 0, 1])

    feat_generator = HierarchicalMotifFeatureGenerator(
            motif_lens=motif_len_vals,
            mutating_positions=mutating_pos_vals,
        )
    max_motif_len = max(motif_len_vals)
    full_motif_dict = feat_generator.feat_gens[-1].motif_dict

    # Load fitted theta file
    with open(args.input_pkl, "r") as f:
        theta = pickle.load(f)[0]

    if len(motif_len_vals) > 1:
        # Combine the hierarchical thetas if that is the case
        full_theta = np.zeros(4**max_motif_len)
        start_idx = 0
        for f in feat_generator.feat_gens[:len(motif_len_vals)]:
            motif_list = f.motif_list
            for m_idx, m in enumerate(motif_list):
                m_theta = theta[start_idx + m_idx]
                if f.offset == 0:
                    full_m_idx = full_motif_dict[m]
                    full_theta[full_m_idx] += m_theta
                else:
                    flanks = itertools.product(["a", "c", "g", "t"], repeat=2*f.offset)
                    for f in flanks:
                        full_m = "".join(f[:f.left_offset]) + m + "".join(f[f.right_offset:])
                        full_m_idx = full_motif_dict[full_m]
                        full_theta[full_m_idx] += m_theta
            start_idx += len(motif_list)
    else:
        full_theta = theta

    convert_to_csv(args.output_csv, full_theta, [max_motif_len], mutating_pos_vals)

    # Call Rscript
    command = 'Rscript'
    script_file = 'R/create_bar_plot_from_file.R'

    cmd = [command, script_file, args.output_csv, str(max_motif_len), args.output_svg]
    print "Calling:", " ".join(cmd)
    res = subprocess.call(cmd)

if __name__ == "__main__":
    main(sys.argv[1:])
