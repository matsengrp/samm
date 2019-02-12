"""
Generate mutated sequences using the survival model on a given tree using
mutation model parameters and partis-generated naive sequences.
NOTE: this script only outputs a pickled tree with the simulated
intermediate/leaf sequences attached and functions differently from other
samm simulation routines.
"""

import argparse
import ete3
import math
import numpy as np
import pickle
import scipy
import subprocess
import sys
import yaml

from simulate_shm_star_tree import create_simulator
from common import *

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed',
        type=int,
        help='Random number generator seed for replicability',
        default=1)
    parser.add_argument('--agg-motif-len',
        type=int,
        help='Length of k-mer motifs in the aggregate model -- assumes that the center position mutates',
        default=5)
    parser.add_argument('--input-tree',
        type=str,
        help='Input newick file with tree',
        default='_output/tree_in.tree')
    parser.add_argument('--output-tree',
        type=str,
        help='Output pickle file with tree',
        default='_output/tree_out.pkl')
    parser.add_argument('--input-model',
        type=str,
        help='Input file with true theta parameters',
        default='_output/true_model.pkl')
    parser.add_argument('--lambda0',
        type=float,
        help='Baseline constant hazard rate in cox proportional hazards model',
        default=0.1)
    parser.add_argument('--with-replacement',
        action="store_true",
        help='Allow same position to mutate multiple times')
    parser.add_argument('--organism',
        type=str,
        help='What species/organism are we simulating for?',
        default='human')
    parser.add_argument('--locus',
        type=str,
        help='What [heavy|light]-chain locus are we simulating for?',
        default='igh')

    parser.set_defaults(with_replacement=False, )
    args = parser.parse_args()
    return args

def simulate_naive_seq(args):
    cmd_str = "partis/bin/partis simulate --outfname simu.yaml --simulate-from-scratch"
    cmd_str += " --seed " + str(args.seed)
    cmd_str += " --species " + args.organism
    cmd_str += " --locus " + args.locus
    subprocess.check_call(cmd_str.split())

    with open("simu.yaml", "rU") as f:
        sim_output = yaml.load(f)

    subprocess.check_call("rm simu.yaml".split())

    return sim_output["events"][0]["naive_seq"]

def run_survival(args, tree, naive_seq):
    simulator = create_simulator(args)

    tree.add_feature("sequence", naive_seq.lower())
    for node in tree.traverse("preorder"):
        sequence = node.sequence
        for children in node.children:
            num_to_mutate = scipy.random.poisson(children.dist * len(sequence) * args.lambda0)
            percent_to_mutate = float(num_to_mutate)/len(sequence)
            full_seq_mutations = simulator.simulate(
                start_seq=sequence,
                percent_mutated=percent_to_mutate,
                with_replacement=args.with_replacement,
            )
            children.add_feature("sequence", full_seq_mutations.end_seq_with_flanks)
    print(tree.get_ascii(attributes=["sequence"], show_internal=True))

def main(args=sys.argv[1:]):
    args = parse_args()
    np.random.seed(args.seed)

    naive_seq = simulate_naive_seq(args)

    subtree = ete3.Tree(args.input_tree)
    # this adds in the root edge (if necessary)
    if math.fabs(subtree.dist - 0.0) <= 1e-10:
        tree = subtree
    else:
        tree = ete3.Tree()
        tree.add_child(subtree)

    run_survival(args, tree, naive_seq)
    with open(args.output_tree, 'wb') as f:
        pickle.dump(tree, f)


if __name__ == "__main__":
    main(sys.argv[1:])
