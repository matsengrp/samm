import sys
import random
import csv
import subprocess

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

def impute_ancestors_dnapars(seqs, gl_seq, gl_name='germline'):
    """
    Compute ancestral states via maximum parsimony

    @param seqs: list of sequences
    @param gl_seq: germline sequence
    @param gl_name: name of germline (must be less than 10 characters long)
    """

    assert(len(gl_name) < 10)

    aln = MultipleSeqAlignment([SeqRecord(Seq(gl_seq), id=gl_name)])

    # sequence ID must be less than ten characters, but also dnapars sets internal node
    # names to 1, 2, 3, ..., so name them numbers descending from 100 million, hoping
    # we won't ever have a clone that big...
    for idx, seq in enumerate(seqs):
        aln.append(SeqRecord(Seq(seq), id=str(99999999-idx)))

    # dnapars uses the name "infile" as default input phylip file
    with open('_output/infile', 'r') as phylip_file:
        phylip_file.write(aln.format('phylip'))

    # and we need to tell it the line where the root sequence occurs
    with open('_output/infile', 'r') as phylip_file:
        for lineno, line in enumerate(phylip_file):
            if line.startswith(gl_name):
                naive_idx = str(lineno)

    # arcane user options for dnapars
    with open('_output/dnapars.config', 'w') as cfg_file:
        cfg_file.write('\n'.join(['O', naive_idx, 'S', 'Y', 'J', '13', '10', '4', '5', '.', 'Y']))

    # dnapars has weird behavior if outfile and outtree already exist o_O
    cmd = ['rm -f _output/outfile _output/outtree']
    print "Calling:", " ".join(cmd)
    res = subprocess.call(cmd, shell=True)

    # defer to command line to construct parsimony trees and ancestral states
    cmd = ['cd _output && dnapars < dnapars.cfg > dnapars.log']
    print "Calling:", " ".join(cmd)
    res = subprocess.call(cmd, shell=True)

    # phew, finally got some trees
    trees = phylip_parse('_output/outfile', gl_name)

    # take first parsimony tree
    root = trees[0].get_tree_root()
    start_seqs = []
    end_seqs = []
    for leaf in root:
        end_seqs.append(leaf.sequence.lower())
        start_seqs.append(leaf.up.sequence.lower())

    return start_seqs, end_seqs

def read_gene_seq_csv_data(gene_file_name, seq_file_name, motif_len=1, impute_ancestors=False, sample_seq=False):
    """
    @param gene_file_name: csv file with germline names and sequences
    @param seq_file_name: csv file with sequence names and sequences, with corresponding germline name
    @param motif_len: length of motif we're using; used to collapse series of "n"s
    @param impute_ancestors: impute ancestors using parsimony
    @param sample_seq: sample one sequence from cluster
    """

    genes = pd.read_csv(gene_file_name)
    seqs = pd.read_csv(seq_file_name)

    full_data = pd.merge(genes, seqs, on='germline_name')

    obs_data = []
    for germline, cluster in full_data.groupby(['germline_name']):
        gl_seq = cluster['germline_sequence'].values[0].lower()
        if impute_ancestors:
            # get parsimony seqs
            proc_gl_seq = re.sub('[^acgtn]', 'n', gl_seq)
            proc_gl_seq = re.sub('^n+|n+$', '', proc_gl_seq)
            seqs_in_cluster = []
            for idx, elt in cluster.iterrows():
                proc_seq = re.sub('[^acgtn]', 'n', elt['sequence'])
                proc_seq = re.sub('^n+|n+$', '', proc_seq)
                if 'n' in proc_seq or len(proc_seq) != len(proc_gl_seq):
                    # for now just ignore sequences with internal "n"s... we'd need
                    # to process all sequences at once (and possibly throw away a lot of
                    # data) if we imputed them for every sequence
                    continue
                seqs_in_cluster.append(proc_seq)

            if not seqs_in_cluster:
                # no sequences after processing
                continue

            start_seqs, end_seqs = impute_ancestors_dnapars(seqs_in_cluster, proc_gl_seq)
        else:
            start_seqs = []
            end_seqs = []
            if sample_seq:
                # choose random sequence
                sampled_index = random.choice(cluster.index)
                row = cluster.loc[sampled_index]
                start_seqs.append(row['germline_sequence'].lower())
                end_seqs.append(row['sequence'].lower())
            else:
                # sample all sequences
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

    return obs_data
