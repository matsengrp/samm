import sys
import random
import csv
import subprocess
import os.path
import pickle
import pandas as pd

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

SAMPLE_RANDOM = 2

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
            germlines[germline_file] = glutils.read_glfo(germline_file, locus='ig'+chain)
    else:
        glfo = glutils.read_glfo(PARTIS_PATH + '/data/germlines/' + species, locus='ig'+chain)
        inferred_gls = [None] * len(annotations_file_names)

    seq_header = ['germline_name', 'sequence_name', 'sequence']

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
                    if len(line['input_seqs']) == 0:
                        # sometimes data will have empty clusters
                        continue
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

def impute_ancestors_dnapars(seqs, gl_seq, scratch_dir, gl_name='germline', verbose=True):
    """
    Compute ancestral states via maximum parsimony

    @param seqs: list of sequences
    @param gl_seq: germline sequence
    @param scratch_dir: where to write intermediate dnapars files
    @param gl_name: name of germline (must be less than 10 characters long)

    @return genes_line: information needed to output imputed germline data
    @return seqs_line: information needed to output imputed sequence data
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
    # 'O', naive_idx: the location of the outgroup root
    # 'S', 'Y': less thorough search; runs much faster but output is less exhaustive
    # 'J', 13, 10: randomize input ("jumble") using seed 13 and jumbling 10 times
    # 4: print out steps in each site (to get all nucleotide info)
    # 5: print sequences in at all nodes (to get ancestors)
    # '.': use dot-differencing for display
    # 'Y': accept these options
    with open(config, 'w') as cfg_file:
        cfg_file.write('\n'.join(['O', naive_idx, 'S', 'Y', 'J', '13', '10', '4', '5', '.', 'Y']))

    # defer to command line to construct parsimony trees and ancestral states
    # dnapars has weird behavior if outfile and outtree already exist o_O
    cmd = ['cd', scratch_dir, '&& rm -f outfile outtree && dnapars <', os.path.basename(config), '> dnapars.log']
    if verbose:
        print "Calling:", " ".join(cmd)
    res = subprocess.call([" ".join(cmd)], shell=True)

    # phew, finally got some trees
    trees = phylip_parse(outfile, gl_name)

    # take first parsimony tree
    genes_line = []
    seq_line = []
    for idx, descendant in enumerate(trees[0].traverse('preorder')):
        if descendant.is_root():
            descendant.name = gl_name
        else:
            # use dummy name for internal node sequences
            descendant.name = '-'.join([descendant.up.name, descendant.name])
            if [descendant.up.name, descendant.up.sequence.lower()] not in genes_line:
                genes_line.append([descendant.up.name, descendant.up.sequence.lower()])
            seq_line.append([descendant.up.name, descendant.name, descendant.sequence.lower()])

    return genes_line, seq_line

def write_data_after_imputing(output_genes, output_seqs, gene_file_name, seq_file_name, motif_len=1, scratch_dir='_output', verbose=True):
    """
    @param output_genes: where to write processed germline data, if wanted
    @param output_genes: where to write processed sequence data, if wanted
    @param gene_file_name: csv file with germline names and sequences
    @param seq_file_name: csv file with sequence names and sequences, with corresponding germline name
    @param motif_len: length of motif we're using; used to collapse series of "n"s
    @param scratch_dir: where to write dnapars intermediate files
    """

    genes = pd.read_csv(gene_file_name)
    seqs = pd.read_csv(seq_file_name)

    full_data = pd.merge(genes, seqs, on='germline_name')

    out_genes = [['germline_name', 'germline_sequence']]
    out_seqs = [['germline_name', 'sequence_name', 'sequence']]
    for gl_idx, (germline, cluster) in enumerate(full_data.groupby(['germline_name'])):
        gl_seq = cluster['germline_sequence'].values[0].lower()
        gl_name = cluster['germline_name'].values[0]
        # Use dnapars to impute nucleotides at intermediate sequences

        # First process sequences to remove unknown nucleotides at the
        # beginning and end of sequences
        proc_gl_seq = re.sub('[^acgtn]', 'n', gl_seq)
        proc_gl_seq = re.sub('^n+|n+$', '', proc_gl_seq)
        seqs_in_cluster = []
        names_in_cluster = []
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
            names_in_cluster.append(elt['sequence_name'])

        if not seqs_in_cluster:
            # No sequences, so dnapars won't do anything, so move on
            continue

        if len(seqs_in_cluster) == 1:
            # If there is only one sequence, dnapars still won't do anything,
            # but there might be information if there are mutations

            if cmp(seqs_in_cluster[0], proc_gl_seq):
                # There are mutations so add to output
                genes_line = [[gl_name, proc_gl_seq]]
                seqs_line = [[gl_name, names_in_cluster[0], seqs_in_cluster[0]]]
            else:
                # No mutations, skip
                continue
        else:
            # otherwise, take it away dnapars
            genes_line, seqs_line = impute_ancestors_dnapars(seqs_in_cluster, proc_gl_seq, scratch_dir, gl_name='gene'+str(gl_idx), verbose=verbose)

        out_genes += genes_line
        out_seqs += seqs_line

    with open(output_genes, 'w') as outg, open(output_seqs, 'w') as outs:
        gene_writer = csv.writer(outg)
        gene_writer.writerows(out_genes)
        seq_writer = csv.writer(outs)
        seq_writer.writerows(out_seqs)

def read_gene_seq_csv_data(gene_file_name, seq_file_name, motif_len=1, sample=1):
    """
    @param gene_file_name: csv file with germline names and sequences
    @param seq_file_name: csv file with sequence names and sequences, with corresponding germline name
    @param motif_len: length of motif we're using; used to collapse series of "n"s
    @param sample: 1: take all sequences; 2: sample random sequence from cluster; 3: choose most highly mutated sequence (default: 1)

    @return ObservedSequenceMutations from processed data
    """

    assert(sample in range(1, 4))

    genes = pd.read_csv(gene_file_name)
    seqs = pd.read_csv(seq_file_name)

    full_data = pd.merge(genes, seqs, on='germline_name')

    obs_data = []
    for gl_idx, (germline, cluster) in enumerate(full_data.groupby(['germline_name'])):
        gl_seq = cluster['germline_sequence'].values[0].lower()
        if sample == 2:
            # Sample a single sequence from a clonal family randomly
            sampled_index = random.choice(cluster.index)
            row = cluster.loc[sampled_index]
            start_seq, end_seq = process_degenerates_and_impute_nucleotides(gl_seq, row['sequence'].lower(), motif_len)

            obs_seq_mutation = ObservedSequenceMutations(
                    start_seq=start_seq,
                    end_seq=end_seq,
                    motif_len=motif_len,
            )

            if obs_seq_mutation.num_mutations > 0:
                # don't consider pairs where mutations occur in flanking regions
                obs_data.append(obs_seq_mutation)
        else:
            for idx, elt in cluster.iterrows():
                n_mutes = 0
                current_obs_seq_mutation = None
                start_seq, end_seq = process_degenerates_and_impute_nucleotides(gl_seq, elt['sequence'].lower(), motif_len)

                obs_seq_mutation = ObservedSequenceMutations(
                        start_seq=start_seq,
                        end_seq=end_seq,
                        motif_len=motif_len,
                )

                if sample == 1 and obs_seq_mutation.num_mutations > 0:
                    # don't consider pairs where mutations occur in flanking regions
                    obs_data.append(obs_seq_mutation)
                elif obs_seq_mutation.num_mutations > n_mutes:
                    current_obs_seq_mutation = obs_seq_mutation

            if sample == 'sample-highly-mutated':
                obs_data.append(current_obs_seq_mutation)

    return obs_data

