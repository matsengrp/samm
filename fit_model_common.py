"""
Some of the shared commands for processing arguments when getting ready to fit models
"""
from common import process_mutating_positions

def process_motif_length_args(args):
    args.motif_lens = [int(m) for m in args.motif_lens.split(',')]

    args.max_motif_len = max(args.motif_lens)

    args.positions_mutating, args.max_mut_pos = process_mutating_positions(args.motif_lens, args.positions_mutating)
    # Find the maximum left and right flanks of the motif with the largest length in the
    # hierarchy in order to process the data correctly
    args.max_left_flank = max(sum(args.positions_mutating, []))
    args.max_right_flank = max([motif_len - 1 - min(left_flanks) for motif_len, left_flanks in zip(args.motif_lens, args.positions_mutating)])

    # Check if our full feature generator will conform to input
    max_left_flanks = args.positions_mutating[args.motif_lens.index(args.max_motif_len)]
    if args.max_left_flank > max(max_left_flanks) or args.max_right_flank > args.max_motif_len - min(max_left_flanks) - 1:
        raise AssertionError('The maximum length motif does not contain all smaller length motifs.')
