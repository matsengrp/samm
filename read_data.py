import sys
import random
import csv
import subprocess
import os.path
import pickle
import pandas as pd
import glob

PARTIS_PATH = './partis'
sys.path.insert(1, PARTIS_PATH + '/python')
from utils import add_implicit_info, process_input_line
import glutils

# needed to read partis files
csv.field_size_limit(sys.maxsize)

from gctree.bin.gctree import phylip_parse

from common import *
from models import ObservedSequenceMutations
from Bio.Seq import Seq
from Bio.SeqIO import SeqRecord
from Bio.AlignIO import MultipleSeqAlignment
from Bio.Phylo.TreeConstruction import ParsimonyScorer, NNITreeSearcher, ParsimonyTreeConstructor
from ete3 import Tree
from itertools import izip

GERMLINE_PARAM_FILE = 'partis/data/germlines/human/igh/ighv.fasta'
SAMPLE_PARTIS_ANNOTATIONS = 'partis/test/reference-results/partition-new-simu-cluster-annotations.csv'

SAMPLE_RANDOM = 2

# TODO: file to convert presto dataset to ours? just correspondence between headers should be enough?
def write_partis_data_from_annotations(output_genes, output_seqs, path_to_annotations, metadata, use_v=True, use_np=False, use_immunized=True, motif_len=1):
    """
    Function to read partis annotations csv

    @param path_to_annotations: path to annotations files
    @param metadata: csv file of metadata; if None defaults will be used for chain/species
    @param use_v: use just the V gene or use the whole sequence?
    @param use_np: use nonproductive sequences only
    @param use_immunized: use immunized mice only
    @param inferred_gls: list of paths to partis-inferred germlines

    @write genes to output_genes and seqs to output_seqs
    """

    partition_info = []
    with open(metadata, 'r') as metafile:
        reader = csv.DictReader(metafile)
        for line in reader:
            annotations = glob.glob(os.path.join(path_to_annotations, 'partitions', line['dataset']+'*-cluster-annotations.csv'))
            if not annotations:
                # no annotations for this dataset (yet?)
                continue
            current_info = {}
            current_info['germline_file'] = os.path.join(path_to_annotations, line['dataset'], 'hmm/germline-sets')
            current_info['annotations_file'] = annotations
            current_info['locus'] = line['locus']
            current_info['species'] = line['species']
            current_info['group'] = line['group']
            if use_immunized and current_info['group'] != 'immunized':
                continue
            current_info['subject'] = line['subject']
            partition_info.append(current_info)

    seq_header = ['germline_name', 'locus', 'clonal_family', 'species', 'group', 'subject', 'sequence_name', 'sequence']

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
        for data_idx, data_info in enumerate(partition_info):
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

                    gl_name = 'clone{}-{}-{}'.format(*[data_idx, idx, line['v_gene']])
                    gene_writer.writerow({'germline_name': gl_name,
                        'germline_sequence': line[gene_col].lower()})

                    for good_idx in good_seq_idx:
                        seq_writer.writerow({'germline_name': gl_name,
                            'locus': data_info['locus'],
                            'clonal_family': gl_name,
                            'species': data_info['species'],
                            'group': data_info['group'],
                            'subject': data_info['subject'],
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
    trees = phylip_parse(outfile, countfile=None, naive=gl_name)

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

    out_genes = []
    out_seqs = []
    for gl_idx, (germline, cluster) in enumerate(full_data.groupby(['germline_name'])):
        seqs_line = []
        genes_line = []
        gl_seq = cluster['germline_sequence'].values[0].lower()
        gl_name = cluster['germline_name'].values[0]
        # Use dnapars to impute nucleotides at intermediate sequences

        # First process sequences to remove unknown nucleotides at the
        # beginning and end of sequences
        proc_gl_seq = re.sub('[^acgtn]', 'n', gl_seq)
        proc_gl_seq = re.sub('^n+|n+$', '', proc_gl_seq)
        seqs_in_cluster = []
        names_in_cluster = []
        meta_in_cluster = cluster.iloc[0].to_dict()
        meta_in_cluster.pop('germline_sequence', None)
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
                genes_line.append({'germline_name': gl_name,
                    'germline_sequence': proc_gl_seq})
                current_seq['germline_name'] = gl_name
                current_seq['sequence_name'] = names_in_cluster[0]
                current_seq['sequence'] = seqs_in_cluster[0]
                seqs_line.append(meta_in_cluster)
            else:
                # No mutations, skip
                continue
        else:
            # otherwise, take it away dnapars
            pars_gene, pars_seq = impute_ancestors_dnapars(
                    seqs_in_cluster,
                    proc_gl_seq,
                    scratch_dir,
                    gl_name='gene'+str(gl_idx),
                    verbose=verbose
                )
            current_seq = meta_in_cluster
            for seq_line in pars_seq:
                current_seq['germline_name'] = seq_line[0]
                current_seq['sequence_name'] = seq_line[1]
                current_seq['sequence'] = seq_line[2]
                seqs_line.append(current_seq)

            for gene_line in pars_gene:
                genes_line.append({'germline_name': gene_line[0],
                    'germline_sequence': gene_line[1]})

        out_genes += genes_line
        out_seqs += seqs_line

    with open(output_genes, 'w') as genes_file, open(output_seqs, 'w') as seqs_file:
        gene_writer = csv.DictWriter(genes_file, list(genes.columns.values))
        gene_writer.writeheader()
        gene_writer.writerows(out_genes)
        seq_writer = csv.DictWriter(seqs_file, list(seqs.columns.values))
        seq_writer.writeheader()
        seq_writer.writerows(out_seqs)

def read_gene_seq_csv_data(
        gene_file_name,
        seq_file_name,
        motif_len=3,
        left_flank_len=1,
        right_flank_len=1,
        sample=1,
        locus='',
        species='',
        ):
    """
    @param gene_file_name: csv file with germline names and sequences
    @param seq_file_name: csv file with sequence names and sequences, with corresponding germline name
    @param motif_len: length of motif we're using; used to collapse series of "n"s
    @param left_flank_len: maximum left flank length for this motif length
    @param right_flank_len: maximum right flank length for this motif length
    @param sample: 1: take all sequences; 2: sample random sequence from cluster; 3: choose most highly mutated sequence (default: 1)
    @param subset_cols: list of names of columns to take subset of data on (e.g., ['chain', 'species'])
    @param subset_vals: list of values of these variables to subset on (e.g., ['k', 'mouse'])

    @return ObservedSequenceMutations from processed data
    """

    assert(sample in range(1, 4))

    genes = pd.read_csv(gene_file_name)
    seqs = pd.read_csv(seq_file_name)
    if locus:
        seqs.where(seqs['locus'] == locus, inplace=True)
    if species:
        seqs.where(seqs['species'] == species, inplace=True)

    full_data = pd.merge(genes, seqs, on='germline_name')

    obs_data = []
    metadata = []
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
                    left_flank_len=left_flank_len,
                    right_flank_len=right_flank_len,
            )

            if obs_seq_mutation.num_mutations > 0:
                # don't consider pairs where mutations occur in flanking regions
                obs_data.append(obs_seq_mutation)
                metadata.append(row)
        else:
            for idx, elt in cluster.iterrows():
                n_mutes = 0
                current_obs_seq_mutation = None
                start_seq, end_seq = process_degenerates_and_impute_nucleotides(gl_seq, elt['sequence'].lower(), motif_len)

                obs_seq_mutation = ObservedSequenceMutations(
                        start_seq=start_seq,
                        end_seq=end_seq,
                        motif_len=motif_len,
                        left_flank_len=left_flank_len,
                        right_flank_len=right_flank_len,
                )

                if sample == 1 and obs_seq_mutation.num_mutations > 0:
                    # don't consider pairs where mutations occur in flanking regions
                    obs_data.append(obs_seq_mutation)
                    metadata.append(elt)
                elif sample == 3 and obs_seq_mutation.num_mutations > n_mutes:
                    current_obs_seq_mutation = obs_seq_mutation

            if sample == 3:
                obs_data.append(current_obs_seq_mutation)
                metadata.append(elt)

    assert(len(obs_data) == len(metadata))

    return obs_data, metadata

def get_data_statistics_print_lines(obs_data, feat_generator):
    """
    Some interesting statistics we can output (some from Cui et al. 2016 for comparison)

    - Number of sequences
    - Number of mutations
    - Number of skipped mutations (bad for us, since these are mutations occurring at flanking areas)
    - Average sequence length
    - Average mutation frequency (avg of n_mutations / seq_len)
    - Number of motifs that had fewer than twenty mutations in center base (bad for Cui because then they average)
    - Number of motifs that had fewer than five hundred mutations in any base (also bad for Cui because then they average again)
    - Number of motifs present in data with any number of mutations

    @return a string that summarizes the data
    """

    n_sequences = len(obs_data)
    total_mutations = 0
    total_skipped_mutations = 0
    avg_seq_len = 0.
    avg_mutation_frequency = 0.
    motif_set = set([])
    mute_set = set([])
    central_base_mutes = [0] * feat_generator.feature_vec_len
    any_mutes = [0] * feat_generator.feature_vec_len
    for obs_seq in obs_data:
        total_mutations += obs_seq.num_mutations
        total_skipped_mutations += obs_seq.skipped_mutations
        avg_seq_len += float(obs_seq.seq_len) / n_sequences
        avg_mutation_frequency += (float(obs_seq.num_mutations) / obs_seq.seq_len) / n_sequences
        motifs = feat_generator.create_for_sequence(obs_seq.start_seq, obs_seq.left_flank, obs_seq.right_flank)
        motif_set.update([item for sublist in motifs.values() for item in sublist])
        # for mutation_pos, _ in obs_seq.mutation_pos_dict.iteritems():
        #     central_base_mutes[motifs[mutation_pos]] += 1
        #     for pos in range(max(mutation_pos-feat_generator.half_motif_len, 0),
        #             min(mutation_pos+feat_generator.half_motif_len+1, obs_seq.seq_len)):
        #         any_mutes[motifs[pos]] += 1

    return '\n'.join([
                '  Number of sequences: %d' % n_sequences,
                '  Number of mutations: %d' % total_mutations,
                '  Number of skipped mutations (flanks): %d' % total_skipped_mutations,
                '  Average sequence length: %f' % avg_seq_len,
                '  Average mutation frequency: %f' % avg_mutation_frequency,
                '  Number of motifs in dataset: %d' % len(motif_set),
                # '  Number of motifs w/ >1 central base mutation: %d' % len([val for val in central_base_mutes if val > 0]),
                # '  Number of motifs w/ <20 mutes in central base: %d' % len([val for val in central_base_mutes if val < 20]),
                # '  Number of motifs w/ <500 mutes in any base: %d' % len([val for val in any_mutes if val < 500]),
            ]
        )
