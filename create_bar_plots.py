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
    parser.add_argument('--positions-mutating',
        type=str,
        help='which position in the motif is mutating',
        default=None)
    parser.add_argument('--output-pdf',
        type=str,
        help='svg file to save output to',
        default='_output/out.pdf')

    args = parser.parse_args()

    return args

def convert_to_csv(target, mutabilities, motif_lens, positions_mutating):
    """
    Take pickle file and convert to csv for use in R
    """
    feat_generator = HierarchicalMotifFeatureGenerator(
            motif_lens=motif_lens,
            left_motif_flank_len_list=positions_mutating,
        )
    motif_list = feat_generator.motif_list
    mutating_pos_list = feat_generator.mutating_pos_list

    positions_mutating = positions_mutating[0]
    motif_len = feat_generator.motif_len
    padded_len = max(positions_mutating)
    min_pad = min(positions_mutating)

    if len(positions_mutating) > 1:
        # pad offset motifs with Ns
        for idx, (motif, mut_pos) in enumerate(zip(motif_list, mutating_pos_list)):
            left_pad = (padded_len - mut_pos) * 'n'
            right_pad = (mut_pos - min_pad) * 'n'
            motif_list[idx] = left_pad + motif + right_pad

    with open(str(target), 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(izip([motif.upper() for motif in motif_list], mutabilities.ravel()))

def plot_theta(args, feat_generator, full_motif_dict, theta, output_pdf):
    if len(args.motif_lens) > 1:
        # Combine the hierarchical thetas if that is the case
        full_theta = np.zeros(4**args.max_motif_len)
        start_idx = 0
        for f in feat_generator.feat_gens[:len(args.motif_lens)]:
            motif_list = f.motif_list
            for m_idx, m in enumerate(motif_list):
                m_theta = theta[start_idx + m_idx]
                if f.hier_offset == 0:
                    full_m_idx = full_motif_dict[m]
                    full_theta[full_m_idx] += m_theta
                else:
                    flanks = itertools.product(["a", "c", "g", "t"], repeat=2*f.hier_offset)
                    for f in flanks:
                        # assume for now hierarchical will just have center mutating
                        full_m = "".join(f[:f.hier_offset]) + m + "".join(f[f.hier_offset:])
                        full_m_idx = full_motif_dict[full_m]
                        full_theta[full_m_idx] += m_theta
            start_idx += len(motif_list)
    else:
        full_theta = theta

    convert_to_csv(args.output_csv, full_theta, [args.max_motif_len], args.positions_mutating)

    # Call Rscript
    command = 'Rscript'
    script_file = 'R/create_bar_plot_from_file.R'

    cmd = [command, script_file, args.output_csv, str(args.max_motif_len), output_pdf]
    print "Calling:", " ".join(cmd)
    res = subprocess.call(cmd)

def main(args=sys.argv[1:]):

    args = parse_args()

    args.motif_lens = [int(m) for m in args.motif_lens.split(',')]
    for m in args.motif_lens:
        assert(m % 2 == 1)
        
    if args.positions_mutating is None:
        # default to central base mutating
        args.max_left_flank = None
        args.max_right_flank = None
        args.positions_mutating = [[m/2] for m in args.motif_lens]
    else:
        args.positions_mutating = [[int(m) for m in positions.split(',')] for positions in args.positions_mutating.split(':')]
        for motif_len, positions in zip(args.motif_lens, args.positions_mutating):
            for m in positions:
                assert(m in range(motif_len))

    feat_generator = HierarchicalMotifFeatureGenerator(
            motif_lens=args.motif_lens,
            left_motif_flank_len_list=args.positions_mutating,
        )
    
    args.max_motif_len = max(args.motif_lens)
    full_motif_dict = feat_generator.feat_gens[-1].motif_dict

    # Load fitted theta file
    with open(args.input_pkl, "r") as f:
        theta = pickle.load(f)[0]

    for col_idx in range(theta.shape[1]):
        output_pdf = args.output_pdf.replace(".pdf", "%d.pdf" % col_idx)
        plot_theta(args, feat_generator, full_motif_dict, theta[:,col_idx], output_pdf)

if __name__ == "__main__":
    main(sys.argv[1:])
