import sys
import argparse
import numpy as np
import os
import os.path
import csv
import warnings

from read_data import *
from scipy.stats import poisson
from ete3 import TreeNode

from submotif_feature_generator import SubmotifFeatureGenerator
from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator

from gctree.bin.mutation_model import MutationModel

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--n-taxa',
        type=int,
        help='number of taxa in each clonal family to simulate; if unspecified, use statistics from partis annotations',
        default=None)
    parser.add_argument('--n-clonal-families',
        type=int,
        help='number of clonal families to generate; if unspecified, use statistics from partis annotations',
        default=None)
    parser.add_argument('--path-to-annotations',
        type=str,
        help='''
        path to partis annotations to determine --n-taxa and/or --n-clonal-families if they are not given
        ''',
        default=None)
    parser.add_argument('--path-to-metadata',
        type=str,
        help='path to partis metadata, same as for annotations',
        default=None)
    parser.add_argument('--seed',
        type=int,
        help='rng seed for replicability',
        default=1533)
    parser.add_argument('--motif-lens',
        type=str,
        default='5',
        help='comma-separated motif lengths (odd only)')
    parser.add_argument('--pct-mutated',
        type=float,
        default=0.025,
        help='what percent of sequence should be mutated')
    parser.add_argument('--output-seqs',
        type=str,
        help='csv file with simulated sequence data',
        default='_output/seqs.csv')
    parser.add_argument('--output-germline-seqs',
        type=str,
        help='csv file with germline genes/nucleotides used',
        default='_output/genes.csv')
    parser.add_argument('--output-per-branch-seqs',
        type=str,
        default='_output/gctree_per_branch_seqs.csv',
        help='csv file for additionally outputting genes from single branches with intermediate ancestors instead of leaves from germline')
    parser.add_argument('--output-per-branch-germline-seqs',
        type=str,
        default='_output/gctree_per_branch_genes.csv',
        help='csv file for additionally outputting genes from single branches with intermediate ancestors instead of leaves from germline')
    parser.add_argument('--max-taxa-per-family',
        type=int,
        default=1000,
        help='the maximum taxa per family to simulate when getting clonal family size statistics')
    parser.add_argument('--mutability',
        type=str,
        default='R/shmulate_params/mut_mouse.csv',
        help='path to mutability model file')
    parser.add_argument('--substitution',
        type=str,
        default='R/shmulate_params/sub_mouse.csv',
        help='path to substitution model file')
    parser.add_argument('--use-v',
        action="store_true",
        help="use V gene only; for computing clonal family statistics")
    parser.add_argument('--use-np',
        action="store_true",
        help="use nonproductive sequences; for computing clonal family statistics")
    parser.add_argument('--use-immunized',
        action="store_true",
        help="use immunized mice; for computing clonal family statistics")
    parser.add_argument('--locus',
        type=str,
        default='',
        help="which locus to use; for computing clonal family statistics")
    parser.add_argument('--branching-parameter',
        type=float,
        default=.9,
        help='poisson branching parameter for gctree')
    parser.add_argument('--min-progeny',
        type=int,
        default=1,
        help='minimum number of progeny for each branch')

    args = parser.parse_args()
    args.motif_lens = sorted(map(int, args.motif_lens.split(",")))

    return args

def create_tree_node(seq, frequency=0):
    tree = TreeNode()
    tree.add_feature('sequence', seq)
    tree.add_feature('frequency', frequency)

    return tree

def generate_singleton_tree(germline_seq, lambda0, mutation_model):
    tree = create_tree_node(germline_seq)
    mutated_seq = mutation_model.mutate(
        germline_seq,
        lambda0=lambda0[0],
        frame=None,
    )
    child = create_tree_node(mutated_seq, frequency=1)
    tree.add_child(child)

    return tree

def run_gctree(args, germline_seq, mutation_model, n_taxa):
    """
    @param args: has branching parameter, minimum number of progeny, percent mutated
                 for simulation settings
    @param germline_seq: starting sequence
    @param mutation_model: gctree MutationModel class with substitution and mutability params
    @param n_taxa: how many taxa to simulate

    @return: ETE tree with MutationModel-simulated data
    """

    mutabilities = mutation_model.mutabilities(germline_seq)
    seq_len = len(germline_seq)
    sequence_mutability = sum(float(mutability[0]) for mutability in mutabilities)/seq_len

    # poisson rate for this sequence (given its relative mutability) is
    #lambda_sequence = sequence_mutability*lambda0
    # so
    #lambda0 = [AVG_NUM_MUTATIONS / sequence_mutability]
    # AVG_NUM_MUTATIONS is % * sequence length divided by the *height*
    # of the tree, which is 1+log2(n_taxa)

    # For one taxon, use mutate; otherwise, use simulate
    lambda0 = [args.pct_mutated * seq_len / (sequence_mutability * (1 + np.log2(n_taxa)))]
    if n_taxa == 1:
        tree = generate_singleton_tree(
            germline_seq,
            lambda0,
            mutation_model,
        )
    else:
        tree = mutation_model.simulate(
            germline_seq,
            lambda0=lambda0,
            N=n_taxa,
            progeny=poisson(
                args.branching_parameter,
                loc=args.min_progeny,
            )
        )

    return tree

def _get_clonal_family_stats(path_to_annotations, metadata, use_np=False, use_immunized=False, locus=''):
    '''
    get data statistics from partis annotations

    @param path_to_annotations: path to partis annotations
    @param metadata: path to partis metadata 
    @param use_np: use nonproductive seqs?
    @param use_immunized: for Cui data, use immunized mice?
    @param locus: which locus to use

    @return list of dicts with clonal family sizes and naive seqs from processed data
    '''

    partition_info = get_partition_info(
        path_to_annotations,
        metadata,
    )

    if use_np:
        # return only nonproductive sequences
        # here "nonproductive" is defined as having a stop codon or being
        # out of frame or having a mutated conserved cysteine
        good_seq = lambda seqs: seqs['stops'] or not seqs['in_frames'] or seqs['mutated_invariants']
    else:
        # return all sequences
        good_seq = lambda seqs: [True for seq in seqs['seqs']]

    all_germline_dicts = []
    for data_idx, data_info in enumerate(partition_info):
        if use_immunized and data_info['group'] != 'immunized':
            continue
        if not locus or data_info['locus'] != locus:
            continue
        PARTIS_PATH = os.path.dirname(os.path.realpath(__file__)) + '/partis'
        sys.path.insert(1, PARTIS_PATH + '/python')
        from utils import add_implicit_info, process_input_line
        import glutils
        glfo = glutils.read_glfo(data_info['germline_file'], locus=data_info['locus'])
        with open(data_info['annotations_file'][0], "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for idx, line in enumerate(reader):
                # add goodies from partis
                if len(line['input_seqs']) == 0:
                    # sometimes data will have empty clusters
                    continue
                process_input_line(line)
                add_implicit_info(glfo, line)
                good_seq_idx = [i for i, is_good in enumerate(good_seq(line)) if is_good]
                if not good_seq_idx:
                    # no nonproductive sequences... skip
                    continue
                else:
                    all_germline_dicts.append({
                        'n_taxa': len(good_seq_idx),
                        'germline_sequence': disambiguate(line['v_gl_seq'].lower()),
                        'germline_name': '-'.join([line['v_gene'], str(idx)]),
                        'v_call': line['v_gene'],
                    })

    return all_germline_dicts

def main(args=sys.argv[1:]):

    args = parse_args()

    # write empty sequence file before appending
    output_dir, _ = os.path.split(args.output_seqs)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Randomly generate number of mutations or use default
    np.random.seed(args.seed)

    all_germline_dicts = _get_clonal_family_stats(
        args.path_to_annotations,
        args.path_to_metadata,
        use_np=args.use_np,
        use_immunized=args.use_immunized,
        locus=args.locus
    )
    if args.n_taxa is None:
        # use number of taxa from dataset
        large_clonal_families = [
            gl_dict['n_taxa'] for gl_dict in all_germline_dicts if gl_dict['n_taxa'] > args.max_taxa_per_family
        ]
        if large_clonal_families:
            warnings.warn("There were {0} clonal families with more than {1} taxa. Ignoring: {2}".format(len(large_clonal_families), args.max_taxa_per_family, large_clonal_families))
            all_germline_dicts = [
                gl_dict for gl_dict in all_germline_dicts if gl_dict['n_taxa'] <= args.max_taxa_per_family
            ]
    else:
        # all clonal families have same number of taxa
        all_germline_dicts = [
            {
            'n_taxa': args.n_taxa,
            'germline_sequence': gl_dict['germline_sequence'],
            'germline_name': gl_dict['germline_name'],
            } for gl_dict in all_germline_dicts
        ]

    if args.n_clonal_families is not None:
        # sample from germline dicts, otherwise take all of them
        all_germline_dicts = np.random.choice(all_germline_dicts, args.n_clonal_families)

    n_unique_naive_seqs = len(set([gl_dict['germline_sequence'] for gl_dict in all_germline_dicts]))
    n_unique_v_calls = len(set([gl_dict['v_call'] for gl_dict in all_germline_dicts]))

    # print clonal family stats
    print "Unique naive seqs: %d" % n_unique_naive_seqs
    print "Unique V calls: %d" % n_unique_v_calls

    mutation_model = MutationModel(args.mutability, args.substitution)

    # Write germline genes to file with two columns: name of gene and
    # corresponding sequence.
    # For each germline gene, run shmulate to obtain mutated sequences.
    # Write sequences to file with three columns: name of germline gene
    # used, name of simulated sequence and corresponding sequence.
    avg_mutations = []
    with open(args.output_seqs, 'w') as out_seqs, open(args.output_germline_seqs, 'w') as out_germline_seqs, \
         open(args.output_per_branch_seqs, 'w') as out_seqs_with_anc, \
         open(args.output_per_branch_germline_seqs, 'w') as out_germline_seqs_with_anc:
        gl_file = csv.writer(out_germline_seqs)
        gl_file.writerow(['germline_name','germline_sequence'])
        gl_anc_file = csv.writer(out_germline_seqs_with_anc)
        gl_anc_file.writerow(['germline_name','germline_sequence'])
        seq_file = csv.writer(out_seqs)
        seq_file.writerow(['germline_name','sequence_name','sequence'])
        seq_anc_file = csv.writer(out_seqs_with_anc)
        seq_anc_file.writerow(['germline_name','sequence_name','sequence'])
        for run, germline_dict in enumerate(all_germline_dicts):
            prefix = "clone%d-" % run
            germline_name = "%s%s" % (prefix, germline_dict['germline_name'])
            # Creates a file with a single run of simulated sequences.
            # The seed is modified so we aren't generating the same
            # mutations on each run
            gl_file.writerow([germline_name, germline_dict['germline_sequence']])
            tree = run_gctree(
                args,
                germline_dict['germline_sequence'].upper(),
                mutation_model,
                germline_dict['n_taxa'],
            )
            random_leaf = random.choice([t.sequence for t in tree.traverse() if t.is_leaf()])
            avg_mutations.append(np.mean([c1 != c2 for c1, c2 in \
                zip(random_leaf.lower(), germline_dict['germline_sequence'].lower())]))

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
                    if descendant.frequency != 0 and descendant.is_leaf() and cmp(descendant.sequence.lower(), germline_dict['germline_sequence']) != 0:
                        # we are at the leaf of the tree and can write into the "observed data" file
                        obs_seq_name = "%s-%s" % (germline_name, seq_name)
                        seq_file.writerow([germline_name, obs_seq_name, descendant.sequence.lower()])

    print "Avg mutations: %.5f" % np.mean(avg_mutations)

if __name__ == "__main__":
    main(sys.argv[1:])
