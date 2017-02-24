import sys
import random

PARTIS_PATH = './partis'
sys.path.insert(1, PARTIS_PATH + '/python')
import utils
import glutils

import sys
import csv
# needed to read partis files
csv.field_size_limit(sys.maxsize)

from common import *
from models import ObservedSequenceMutations

GERMLINE_PARAM_FILE = PARTIS_PATH + '/data/germlines/human/h/ighv.fasta'
SAMPLE_PARTIS_ANNOTATIONS = PARTIS_PATH + '/test/reference-results/partition-new-simu-cluster-annotations.csv'

# TODO: file to convert presto dataset to ours? just correspondence between headers should be enough?
def write_partis_data_from_annotations(output_genes, output_seqs, annotations_file_names, chain='h', use_v=True, species='human', use_np=True, inferred_gls=None, motif_len=1, output_presto=False, impute_ancestors=False):
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

    if output_presto:
        seq_header = utils.presto_headers.values()
    else:
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
                    filtered_line = line.copy()
                    if impute_ancestors:
                        # dress each sequence with its ancestor and put into column gene_col using all seqs
                        # then filter based on good_seq()
                        filtered_line = compute_ancestors(filtered_line, gene_col, seqs_col)
                        for col in [gene_col, seqs_col, 'unique_ids']:
                            filtered_line[col] = []
                            for good_idx in good_seq_idx:
                                filtered_line[col].append(line[col][good_idx])
                        # TODO: this will repeat some germlines since they're shared between sequences... oh well
                    else:
                        # otherwise pick a random good_seq()
                        random_idx = random.choice(good_seq_idx)
                        for col in [seqs_col, 'unique_ids']:
                            filtered_line[col] = line[col][random_idx]
                        gl_name = 'clone{}-{}-{}'.format(*[data_idx, idx, line['v_gene']])
                        gene_writer.writerow({'germline_name': gl_name,
                            'germline_sequence': filtered_line[gene_col].lower()})
                        seq_writer.writerow(get_seq_line(filtered_line, seqs_col, gl_name, output_presto))

def get_seq_line(line, seqs_col, gl_name, output_presto):
    """
    @param lines: partis sequence/germline data
    @param output_presto: whether to output presto data
    @param gene_col: column of lines where germline gene sits
    @param seqs_col: column of lines where sequence sits

    @return list of dicts containing sequences and ancestors
    """

    if output_presto:
        # output with presto data/headers
        datum = utils.convert_to_presto_headers(line)
    else:
        # output with our data/headers
        datum = {'germline_name': gl_name,
            'sequence_name': '-'.join([gl_name, line['unique_ids']]),
            'sequence': line[seqs_col].lower()}

    return datum

def compute_ancestors(line, gene_col, seqs_col):
    """
    Compute ancestral states via maximum parsimony
    """

    return line

def read_gene_seq_csv_data(gene_file_name, seq_file_name, motif_len=1):
    """
    @param gene_file_name: csv file with germline names and sequences
    @param seq_file_name: csv file with sequence names and sequences, with corresponding germline name
    @param motif_len: length of motif we're using; used to collapse series of "n"s
    """
    gene_dict = {}
    with open(gene_file_name, "r") as gene_csv:
        gene_reader = csv.reader(gene_csv, delimiter=',')
        gene_reader.next()
        for row in gene_reader:
            gene_dict[row[0]] = row[1].lower()

    obs_data = []
    with open(seq_file_name, "r") as seq_csv:
        seq_reader = csv.reader(seq_csv, delimiter=",")
        seq_reader.next()
        for row in seq_reader:
            # process sequences
            start_seq, end_seq = process_degenerates_and_impute_nucleotides(gene_dict[row[0]], row[2].lower(), motif_len)
            obs_data.append(
                ObservedSequenceMutations(
                    start_seq=start_seq,
                    end_seq=end_seq,
                    motif_len=motif_len,
                )
            )
    return gene_dict, obs_data
