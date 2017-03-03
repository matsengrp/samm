import sys
import random
import csv
import subprocess
import os.path
import pickle

PARTIS_PATH = './partis'
sys.path.insert(1, PARTIS_PATH + '/python')
import utils
import glutils

# needed to read partis files
csv.field_size_limit(sys.maxsize)

sys.path.append('./gctree/bin')
from gctree import phylip_parse

from common import *
from models import ObservedSequenceMutations
from Bio.Seq import Seq
from Bio.SeqIO import SeqRecord
from Bio.AlignIO import MultipleSeqAlignment
from Bio.Phylo.TreeConstruction import ParsimonyScorer, NNITreeSearcher, ParsimonyTreeConstructor
from ete3 import Tree
from itertools import izip

GERMLINE_PARAM_FILE = PARTIS_PATH + '/data/germlines/human/h/ighv.fasta'
SAMPLE_PARTIS_ANNOTATIONS = PARTIS_PATH + '/test/reference-results/partition-new-simu-cluster-annotations.csv'

# TODO: file to convert presto dataset to ours? just correspondence between headers should be enough?
def write_partis_data_from_annotations(output_genes, output_seqs, annotations_file_names, chain='h', use_v=True, species='human', use_np=True, inferred_gls=None, motif_len=1):
    """
    Function to read partis annotations csv

    @param annotations_file_names: list of paths to annotations files
    @param chain: h for heavy, k or l for kappa or lambda light chain
    @param use_v: use just the V gene or use the whole sequence?
    @param species: 'human' or 'mouse'
    @param use_np: use nonproductive sequences only
    @param inferred_gls: list of paths to partis-inferred germlines

    @write genes to output_genes and seqs to output_seqs
    """

    if not isinstance(annotations_file_names, list):
        annotations_file_names = [annotations_file_names]

    # read default germline info
    if inferred_gls is not None:
        if not isinstance(inferred_gls, list):
            inferred_gls = [inferred_gls]
        germlines = {}
        for germline_file in set(inferred_gls):
            germlines[germline_file] = glutils.read_glfo(germline_file, chain=chain)
    else:
        glfo = glutils.read_glfo(PARTIS_PATH + '/data/germlines/' + species, chain=chain)
        inferred_gls = [None] * len(annotations_file_names)

    seq_header = ['germline_name', 'sequence_name', 'sequence']

    obs_data = pd.DataFrame(columns=seq_header)

    seqs_col = 'v_qr_seqs' if use_v else 'seqs'
    gene_col = 'v_gl_seq' if use_v else 'naive_seq'

    if use_np:
        # return only nonproductive sequences
        # here "nonproductive" is defined as having a stop codon or being
        # out of frame or having a mutated conserved cysteine
        good_seq = lambda seqs: seqs['stops'] or not seqs['in_frames'] or seqs['mutated_invariants']
    else:
        # return all sequences
        good_seq = lambda seqs: [True for seq in seqs[seqs_col]]

    with open(output_genes, 'w') as genes_file, open(output_seqs, 'w') as seqs_file:
        gene_writer = csv.DictWriter(genes_file, ['germline_name', 'germline_sequence'])
        gene_writer.writeheader()
        seq_writer = csv.DictWriter(seqs_file, seq_header)
        seq_writer.writeheader()
        for data_idx, (annotations_file, germline_file) in enumerate(zip(annotations_file_names, inferred_gls)):
            if germline_file is not None:
                glfo = germlines[germline_file]
            with open(annotations_file, "r") as csvfile:
                reader = csv.DictReader(csvfile)
                for idx, line in enumerate(reader):
                    # add goodies from partis
                    utils.process_input_line(line)
                    utils.add_implicit_info(glfo, line)
                    good_seq_idx = [i for i, is_good in enumerate(good_seq(line)) if is_good]
                    if not good_seq_idx:
                        # no nonproductive sequences... skip
                        continue

                    gl_name = 'clone{}-{}-{}'.format(*[data_idx, idx, line['v_gene']])
                    gene_writer.writerow({'germline_name': gl_name,
                        'germline_sequence': line[gene_col].lower()})

                    for good_idx in good_seq_idx:
                        seq_writer.writerow({'germline_name': gl_name,
                            'sequence_name': '-'.join([gl_name, line['unique_ids'][good_idx]]),
                            'sequence': line[seqs_col][good_idx].lower()})

def impute_ancestors_dnapars(seqs, gl_seq, scratch_dir, gl_name='germline'):
    """
    Compute ancestral states via maximum parsimony

    @param seqs: list of sequences
    @param gl_seq: germline sequence
    @param gl_name: name of germline (must be less than 10 characters long)
    """

    assert(len(gl_name) < 10)

    infile, config, outfile = [
            os.path.join(scratch_dir, fname) for fname in [
                'infile',
                'dnapars.cfg',
                'outfile',
                ]
            ]

    aln = MultipleSeqAlignment([SeqRecord(Seq(gl_seq), id=gl_name)])

    # sequence ID must be less than ten characters, but also dnapars sets internal node
    # names to 1, 2, 3, ..., so name them numbers descending from 100 million, hoping
    # we won't ever have a clone that big...
    for idx, seq in enumerate(seqs):
        aln.append(SeqRecord(Seq(seq), id=str(99999999-idx)))

    # dnapars uses the name "infile" as default input phylip file
    with open(infile, 'w') as phylip_file:
        phylip_file.write(aln.format('phylip'))

    # and we need to tell it the line where the root sequence occurs
    with open(infile, 'r') as phylip_file:
        for lineno, line in enumerate(phylip_file):
            if line.startswith(gl_name):
                naive_idx = str(lineno)

    # arcane user options for dnapars
    with open(config, 'w') as cfg_file:
        cfg_file.write('\n'.join(['O', naive_idx, 'J', '13', '10', '4', '5', '.', 'Y']))

    # defer to command line to construct parsimony trees and ancestral states
    # dnapars has weird behavior if outfile and outtree already exist o_O
    cmd = ['cd', scratch_dir, '&& rm -f outfile outtree && dnapars <', os.path.basename(config), '> dnapars.log']
    print "Calling:", " ".join(cmd)
    res = subprocess.call([" ".join(cmd)], shell=True)

    # phew, finally got some trees
    trees = phylip_parse(outfile, gl_name)

    # take first parsimony tree
    root = trees[0].get_tree_root()
    start_seqs = []
    end_seqs = []
    for leaf in root:
        end_seqs.append(leaf.sequence.lower())
        start_seqs.append(leaf.up.sequence.lower())

    return start_seqs, end_seqs

def read_gene_seq_csv_data(gene_file_name, seq_file_name, motif_len=1, sample_or_impute=None, scratch_dir='_output', output_file=None):
    """
    @param gene_file_name: csv file with germline names and sequences
    @param seq_file_name: csv file with sequence names and sequences, with corresponding germline name
    @param motif_len: length of motif we're using; used to collapse series of "n"s
    @param impute_ancestors: impute ancestors using parsimony
    @param sample_seq: sample one sequence from cluster
    """

    assert(sample_or_impute in [None, 'impute-ancestors', 'sample-random', 'sample-highly-mutated'])

    genes = pd.read_csv(gene_file_name)
    seqs = pd.read_csv(seq_file_name)

    full_data = pd.merge(genes, seqs, on='germline_name')

    obs_data = []
    for germline, cluster in full_data.groupby(['germline_name']):
        gl_seq = cluster['germline_sequence'].values[0].lower()
        if sample_or_impute == 'impute-ancestors':
            # Use dnapars to impute nucleotides at intermediate sequences

            # First process sequences to remove unknown nucleotides at the
            # beginning and end of sequences
            proc_gl_seq = re.sub('[^acgtn]', 'n', gl_seq)
            proc_gl_seq = re.sub('^n+|n+$', '', proc_gl_seq)
            seqs_in_cluster = []
            for idx, elt in cluster.iterrows():
                proc_seq = re.sub('[^acgtn]', 'n', elt['sequence'])
                proc_seq = re.sub('^n+|n+$', '', proc_seq)
                if 'n' in proc_seq or len(proc_seq) != len(proc_gl_seq):
                    # If a sequence has internal "n"s, we would need to
                    # propagate that to all sequences for our processing of
                    # dnapars to work, which may throw away too much data.
                    # Instead throw away that sequence...
                    continue
                seqs_in_cluster.append(proc_seq)

            if not seqs_in_cluster:
                # If there are no sequences after processing then dnapars
                # won't do anything, so move on
                continue

            start_seqs, end_seqs = impute_ancestors_dnapars(seqs_in_cluster, proc_gl_seq, scratch_dir)
        elif sample_or_impute == 'sample-random':
            # Sample a single sequence from a clonal family randomly
            sampled_index = random.choice(cluster.index)
            row = cluster.loc[sampled_index]
            start_seqs = [gl_seq]
            end_seqs = [row['sequence'].lower()]
        elif sample_or_impute == 'sample-highly-mutated':
            # Choose the sequence from a clonal family that has the highest number
            # of mutations
            n_mutes = 0
            end_seqs = [gl_seq]
            start_seqs = [gl_seq]
            for idx, elt in cluster.iterrows():
                current_mutes = sum([c0 != c1 for c0, c1 in izip(gl_seq, elt['sequence'].lower())])
                if current_mutes > n_mutes:
                    end_seqs = [elt['sequence'].lower()]
        else:
            # Take all sequences as-is
            start_seqs = []
            end_seqs = []
            for idx, elt in cluster.iterrows():
                start_seqs.append(gl_seq)
                end_seqs.append(elt['sequence'].lower())

        for start_seq, end_seq in zip(start_seqs, end_seqs):
            # process sequences
            start_seq, end_seq = process_degenerates_and_impute_nucleotides(start_seq, end_seq, motif_len)

            obs_seq_mutation = ObservedSequenceMutations(
                    start_seq=start_seq,
                    end_seq=end_seq,
                    motif_len=motif_len,
            )

            if obs_seq_mutation.num_mutations > 0:
                # mutations in nonflanking region, so don't skip
                obs_data.append(obs_seq_mutation)

    if output_file is not None:
        # write to pickled file to save on computation
        pickle.dump(obs_data, open(output_file, 'w'))

    return obs_data
