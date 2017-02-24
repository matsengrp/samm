import sys
import random

PARTIS_PATH = './partis'
sys.path.insert(1, PARTIS_PATH + '/python')
import utils
import glutils

import sys
import csv

PARTIS_PATH = './partis'
sys.path.insert(1, PARTIS_PATH + '/python')
import utils
import glutils

# needed to read partis files
csv.field_size_limit(sys.maxsize)

from common import *
from models import ObservedSequenceMutations

SAMPLE_PARTIS_ANNOTATIONS = PARTIS_PATH + '/test/reference-results/partition-new-simu-cluster-annotations.csv'

def read_partis_annotations(annotations_file_names, chain='h', use_v=True, species='human', use_np=True, inferred_gls=None, motif_len=1):
    """
    Function to read partis annotations csv

    @param annotations_file_names: list of paths to annotations files
    @param chain: h for heavy, k or l for kappa or lambda light chain
    @param use_v: use just the V gene or use the whole sequence?
    @param species: 'human' or 'mouse'
    @param use_np: use nonproductive sequences only
    @param inferred_gls: list of paths to partis-inferred germlines

    TODO: do we want to output intermediate genes.csv/seqs.csv files?

    TODO: do we want support for including multiple annotations files?
    is this something people do, or is this done at the partis level?

    @return gene_dict, obs_data
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

    gene_dict = {}
    obs_data = []

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

    for annotations_file, germline_file in zip(annotations_file_names, inferred_gls):
        if germline_file is not None:
            glfo = germlines[germline_file]
        with open(annotations_file, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for idx, line in enumerate(reader):
                # add goodies from partis
                utils.process_input_line(line)
                utils.add_implicit_info(glfo, line)
                # for now just use V gene for ID
                key = 'clone{}-{}'.format(*[idx, line['v_gene']])
                gene_dict[key] = line[gene_col].lower()
                good_seqs = [seq for seq, cond in zip(line[seqs_col], good_seq(line)) if cond]
                end_seq = random.choice(good_seqs)
                # process sequences
                gl_seq, ch_seq = process_degenerates_and_impute_nucleotides(line[gene_col].lower(), end_seq.lower(), motif_len)
                obs_data.append(
                    ObservedSequenceMutations(
                        start_seq=gl_seq,
                        end_seq=ch_seq,
                        motif_len=motif_len,
                    )
                )
    return gene_dict, obs_data


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
            if cmp(start_seq, end_seq) == 0:
                # Sequences are the same therefore no mutations, so skip this entry
                continue
            obs_data.append(
                ObservedSequenceMutations(
                    start_seq=start_seq,
                    end_seq=end_seq,
                    motif_len=motif_len,
                )
            )
    return gene_dict, obs_data
