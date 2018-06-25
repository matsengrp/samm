import sys
import random
import csv
import subprocess
import os.path
import pickle
import pandas as pd
import glob
import copy

from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator
from motif_feature_generator import MotifFeatureGenerator


# needed to read partis files
csv.field_size_limit(sys.maxsize)

from common import *
from models import ObservedSequenceMutations
from Bio.Seq import Seq
from Bio.SeqIO import SeqRecord
from Bio.AlignIO import MultipleSeqAlignment
from ete3 import Tree
from itertools import izip

GERMLINE_PARAM_FILE = 'partis/data/germlines/human/igh/ighv.fasta'
SAMPLE_PARTIS_ANNOTATIONS = 'partis/test/reference-results/partition-new-simu-cluster-annotations.csv'

SAMPLE_RANDOM = 2

def read_zero_motif_csv(csv_file_name, per_target_model):
    """
    Reads a csv file that indicates which motif (and target nucleotides) are zero constants
    The csv file format should be:
        lowercase motif, mutating position, bool(theta val is zero for col i) for i in range(theta.shape[1])
    Example:
        aaaaa, 2, 1, 0, 1, 1, 0
        means "aaaaa" central base to anything has theta value zero,
                "aaaaa" central base to "a" has theta value anything (actually will be set to -inf),
                "aaaaa" central base to "c" has theta value zero,
                "aaaaa" central base to "g" has theta value zero,
                "aaaaa" central base to "t" has theta value anything

    @return motifs_to_remove: motifs that should be completely removed from the entire MCMC-EM procedure
                                since all theta values associated with that motif is zero
    @return target_pairs_to_remove: dictionary containing motifs with some zero theta values
                                    the value of the dictionary is a list of all targets with theta value zero
                                    (for any-target theta, we indicate it with an "n" - it corresponds to col 0)
    """
    motifs_to_remove = []
    pos_to_remove = []
    target_pairs_to_remove = dict()

    if csv_file_name == "":
        return motifs_to_remove, pos_to_remove, target_pairs_to_remove

    with open(csv_file_name, "r") as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            motif = row[0]
            mut_pos = row[1]
            zero_thetas = []
            if not per_target_model:
                if int(row[2]) == 1:
                    motifs_to_remove.append(motif)
                    pos_to_remove.append(mut_pos)
            else:
                if int(row[2]) == 1:
                    zero_thetas.append("n")
                for i in range(NUM_NUCLEOTIDES):
                    if int(row[3 + i]) == 1:
                        zero_thetas.append(NUCLEOTIDES[i])
                if len(zero_thetas) >= NUM_NUCLEOTIDES:
                    motifs_to_remove.append(motif)
                    pos_to_remove.append(mut_pos)
                else:
                    target_pairs_to_remove[motif] = dict()
                    target_pairs_to_remove[motif][mut_pos] = zero_thetas
    return motifs_to_remove, pos_to_remove, target_pairs_to_remove

def get_partition_info(path_to_annotations, metadata):
    """
    Process partis annotations to obtain info from partitioned data

    @return partition_info: list of dictionaries containing various bits of information about each cluster
    """
    partition_info = []
    with open(metadata, 'r') as metafile:
        reader = csv.DictReader(metafile)
        for line in reader:
            annotations = os.path.join(
                path_to_annotations,
                'partitions',
                '-'.join([line['dataset'], 'cluster-annotations.csv']),
            )
            if not annotations:
                # no annotations for this dataset
                continue
            line['germline_file'] = os.path.join(
                path_to_annotations,
                line['dataset'],
                'hmm/germline-sets'
            )
            line['annotations_file'] = annotations
            partition_info.append(line)

    return partition_info

def write_partis_data_from_annotations(
    output_genes,
    output_seqs,
    path_to_annotations,
    metadata,
    filters={},
    seq_filters={},
    min_clonal_family_size=0,
    min_seq_len=0,
    max_mut_pct=1.,
    min_mut_pct=0.,
    clone_str='',
    region='v',
    germline_family='v',
):
    """
    Function to read partis annotations csv

    @param path_to_annotations: path to annotations files
    @param metadata: csv file of metadata; if None defaults will be used for chain/species
    @param filters: dictionary of lists with keys as column name and items as those values of the column variable to retain;
        filters out families, e.g., {'locus': ['igk']}, etc.
    @param seq_filters: same as filters, but for sequences, e.g., {indel_reversed_seqs': [''], 'in_frames': [False]} will
        only retain sequences that are out of frame and did not have an indel
    @param min_clonal_family_size: minimum clonal family size
    @param min_seq_len: minimum sequence length
    @param max_mut_pct: maximum mutation percentage
    @param min_mut_pct: minimum mutation percentage
    @param clone_str: string for identifying clones (useful if merging annotations from multiple datasets)
    @param region: B-cell receptor region ('v', 'd', 'j', or 'vdj')
    @param germline_family: for performing cross validation ('v', 'd', or 'j')

    @write genes to output_genes and seqs to output_seqs
    """

    families = ['v', 'd', 'j']
    if germline_family not in families:
        raise ValueError("Invalid germline_family: %s. Must be one of %s" % (germline_family, families))

    regions = ['v', 'd', 'j', 'vdj']
    if region not in regions:
        raise ValueError("Invalid region: %s. Must be one of %s" % (region, regions))

    PARTIS_PATH = os.path.dirname(os.path.realpath(__file__)) + '/partis'
    sys.path.insert(1, PARTIS_PATH + '/python')
    from utils import add_implicit_info, process_input_line
    import glutils

    partition_info = get_partition_info(
        path_to_annotations,
        metadata,
    )

    with open(output_genes, 'w') as genes_file, open(output_seqs, 'w') as seqs_file:
        gene_writer = csv.DictWriter(genes_file, ['germline_name', 'germline_sequence'])
        gene_writer.writeheader()

        seq_header = [
            'germline_name',
            'sequence_name',
            'sequence',
            'germline_family',
            'v_gene',
            'region',
        ]

        for key, _ in filters.iteritems():
            seq_header += [key]

        seq_writer = csv.DictWriter(seqs_file, seq_header)
        seq_writer.writeheader()
        for data_idx, data_info in enumerate(partition_info):
            if any([data_info[key] not in values for key, values in filters.iteritems()]):
                continue
            glfo = glutils.read_glfo(data_info['germline_file'], locus=data_info['locus'])
            with open(data_info['annotations_file'], "r") as csvfile:
                reader = csv.DictReader(csvfile)
                for idx, line in enumerate(reader):
                    if line['v_gene'] == '':
                        # failed annotations
                        continue

                    # add goodies from partis
                    process_input_line(line)
                    add_implicit_info(glfo, line)
                    n_seqs = len(line['input_seqs'])
                    if n_seqs < min_clonal_family_size:
                        # don't take small clonal families---for data quality purposes
                        continue

                    if region == 'vdj':
                        gl_seq = line['naive_seq'].lower()
                        all_seqs = [seq.lower() for seq in line['seqs']]
                    else:
                        gl_seq = line['v_gl_seq'].lower()
                        all_seqs = [seq.lower() for seq in line['v_qr_seqs']]

                    idx_list = []
                    # frequency filter
                    idx_list.append(set([i for i, val in enumerate(line['mut_freqs']) if val < max_mut_pct and val >= min_mut_pct]))
                    # sequence length filter
                    idx_list.append(set([i for i, val in enumerate(all_seqs) if len(val.translate(None, 'n')) > min_seq_len]))
                    for key, values in seq_filters.iteritems():
                        idx_list.append(set([i for i, val in enumerate(line[key]) if val in values]))

                    good_seq_idx = set.intersection(*idx_list)
                    if not good_seq_idx:
                        # no sequences after filtering... skip
                        continue

                    gl_name = 'clone{}-{}-{}'.format(*[data_idx, idx, clone_str])
                    gene_writer.writerow({
                        'germline_name': gl_name,
                        'germline_sequence': gl_seq,
                    })

                    for good_idx in good_seq_idx:
                        base_dict = {
                            'germline_name': gl_name,
                            'sequence_name': '-'.join([gl_name, line['unique_ids'][good_idx]]),
                            'sequence': all_seqs[good_idx].lower(),
                            'germline_family': line['{}_gene'.format(germline_family)][:5],
                            'v_gene': line['v_gene'],
                            'region': region,
                        }

                        for key, _ in filters.iteritems():
                            base_dict[key] = data_info[key]

                        seq_writer.writerow(base_dict)

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
    from gctree.bin.phylip_parse import parse_outfile

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
    trees = parse_outfile(outfile, countfile=None, naive=gl_name)

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

def disambiguate(seq):
    """
    @param seq: sequence

    @return sequence where only unknown nucleotides are "n"s
    """
    return re.sub('[^acgtn]', 'n', seq)

def write_data_after_sampling(output_genes, output_seqs, gene_file_name, seq_file_name, sample_highest_mutated=False):
    """
    @param output_genes: where to write processed germline data, if wanted
    @param output_genes: where to write processed sequence data, if wanted
    @param gene_file_name: csv file with germline names and sequences
    @param seq_file_name: csv file with sequence names and sequences, with corresponding germline name
    @param sample_highest_mutated: sample sequence from each clonal family with most mutations
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
        proc_gl_seq = disambiguate(gl_seq)
        if sample_highest_mutated:
            max_mutations = 0
            # choose at random if they all have same number of mutations
            sampled_indices = [random.choice(cluster.index)]
            for idx, elt in cluster.iterrows():
                proc_seq = disambiguate(elt['sequence'].lower())
                current_num_mutations = sum([c1 != c2 for c1, c2 in zip(proc_seq, proc_gl_seq)])
                if len(proc_seq) != len(proc_gl_seq):
                    continue
                elif current_num_mutations > max_mutations:
                    sampled_indices = [idx]
                    max_mutations = current_num_mutations
                elif current_num_mutations == max_mutations:
                    sampled_indices.append(idx)
            sampled_index = random.choice(sampled_indices)
        else:
            sampled_index = random.choice(cluster.index)

        elt = cluster.loc[sampled_index]

        meta_in_cluster = cluster.iloc[0].to_dict()
        meta_in_cluster.pop('germline_sequence', None)

        proc_seq = disambiguate(elt['sequence'].lower())

        current_seq = meta_in_cluster.copy()
        if cmp(proc_seq, proc_gl_seq):
            # There are mutations so add to output
            genes_line.append({
                'germline_name': gl_name,
                'germline_sequence': proc_gl_seq,
            })
            current_seq['germline_name'] = gl_name
            current_seq['sequence_name'] = elt['sequence_name']
            current_seq['sequence'] = proc_seq
            seqs_line.append(current_seq)
        else:
            # No mutations, skip
            continue

        out_genes += genes_line
        out_seqs += seqs_line

    with open(output_genes, 'w') as genes_file, open(output_seqs, 'w') as seqs_file:
        gene_writer = csv.DictWriter(genes_file, list(genes.columns.values))
        gene_writer.writeheader()
        gene_writer.writerows(out_genes)
        seq_writer = csv.DictWriter(seqs_file, list(seqs.columns.values))
        seq_writer.writeheader()
        seq_writer.writerows(out_seqs)

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

            current_seq = meta_in_cluster.copy()
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
            gl_name = 'gene'+str(gl_idx)
            pars_gene, pars_seq = impute_ancestors_dnapars(
                    seqs_in_cluster,
                    proc_gl_seq,
                    scratch_dir,
                    gl_name=gl_name,
                    verbose=verbose
                )
            for seq_line in pars_seq:
                current_seq = meta_in_cluster.copy()
                current_seq['germline_name'] = gl_name
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

def get_sequence_mutations_from_tree(tree, motif_len=5, left_flank_len=None, right_flank_len=None):
    """
    Given an ETE tree, return a list of observed sequence mutations
    """
    if left_flank_len is None or right_flank_len is None:
        left_flank_len = motif_len/2
        right_flank_len = motif_len/2

    obs_data = []
    for _, descendant in enumerate(tree.traverse('preorder')):
        if not descendant.is_root():
            start_seq, end_seq, collapse_list = process_degenerates_and_impute_nucleotides(
                descendant.up.sequence.lower(),
                descendant.sequence.lower(),
                motif_len
            )

            obs_seq_mutation = ObservedSequenceMutations(
                    start_seq=start_seq,
                    end_seq=end_seq,
                    motif_len=motif_len,
                    left_flank_len=left_flank_len,
                    right_flank_len=right_flank_len,
                    collapse_list=collapse_list,
            )

            if obs_seq_mutation.num_mutations > 0:
                # don't consider pairs where mutations occur in flanking regions
                obs_data.append(obs_seq_mutation)

    return obs_data

def read_gene_seq_csv_data(
        gene_file_name,
        seq_file_name,
        motif_len=3,
        left_flank_len=None,
        right_flank_len=None,
        ):
    """
    @param gene_file_name: csv file with germline names and sequences
    @param seq_file_name: csv file with sequence names and sequences, with corresponding germline name
    @param motif_len: length of motif we're using; used to collapse series of "n"s
    @param left_flank_len: maximum left flank length for this motif length
    @param right_flank_len: maximum right flank length for this motif length

    @return ObservedSequenceMutations from processed data
    """

    if left_flank_len is None or right_flank_len is None:
        # default to central base mutating
        left_flank_len = motif_len/2
        right_flank_len = motif_len/2

    genes = pd.read_csv(gene_file_name)
    seqs = pd.read_csv(seq_file_name)

    full_data = pd.merge(genes, seqs, on='germline_name')

    obs_data = []
    metadata = []
    for gl_idx, (germline, cluster) in enumerate(full_data.groupby(['germline_name'])):
        gl_seq = cluster['germline_sequence'].values[0].lower()
        for idx, elt in cluster.iterrows():
            n_mutes = 0
            current_obs_seq_mutation = None
            start_seq, end_seq, collapse_list = process_degenerates_and_impute_nucleotides(gl_seq, elt['sequence'].lower(), motif_len)

            obs_seq_mutation = ObservedSequenceMutations(
                    start_seq=start_seq,
                    end_seq=end_seq,
                    motif_len=motif_len,
                    left_flank_len=left_flank_len,
                    right_flank_len=right_flank_len,
                    collapse_list=collapse_list,
            )

            if obs_seq_mutation.num_mutations > 0:
                # don't consider pairs where mutations occur in flanking regions
                obs_data.append(obs_seq_mutation)
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
    seq_lens = []
    avg_mutation_frequency = 0.
    motif_set = set([])
    mute_set = set([])
    central_base_mutes = [0] * feat_generator.feature_vec_len
    any_mutes = [0] * feat_generator.feature_vec_len
    for obs_seq in obs_data:
        total_mutations += obs_seq.num_mutations
        total_skipped_mutations += obs_seq.skipped_mutations
        seq_lens.append(obs_seq.seq_len)
        avg_mutation_frequency += (float(obs_seq.num_mutations) / obs_seq.seq_len) / n_sequences
        motifs = feat_generator.create_for_sequence(obs_seq.start_seq, obs_seq.left_flank, obs_seq.right_flank)
        motif_set.update([item for sublist in motifs.values() for item in sublist])

    return '\n'.join([
                '  Number of sequences: %d' % n_sequences,
                '  Number of mutations: %d' % total_mutations,
                '  Number of skipped mutations (flanks): %d' % total_skipped_mutations,
                '  Median sequence length: %d' % int(np.median(seq_lens)),
                '  Average mutation frequency: %.2f' % (avg_mutation_frequency * 100),
                '  Number of motifs in dataset in the germline sequences: %d' % len(motif_set),
            ]
        )

def load_true_model(file_name):
    with open(file_name, "r") as f:
        true_model_agg, true_model = pickle.load(f)
    return np.array(true_model_agg)

def load_fitted_model(file_name, keep_col0=False, add_targets=True):
    with open(file_name, "r") as f:
        fitted_models = pickle.load(f)
        best_model = pick_best_model(fitted_models)

    if best_model is None:
        print "FAIL", file_name
        return None

    hier_feat_gen = best_model.refit_feature_generator
    best_model.agg_refit_theta = hier_feat_gen.create_aggregate_theta(
        best_model.refit_theta,
        keep_col0=keep_col0,
        add_targets=add_targets,
    )

    # for penalized theta create full feature generator with no features removed
    full_feat_gen = copy.deepcopy(hier_feat_gen)
    full_feat_gen.update_feats_after_removing(None)
    best_model.agg_penalized_theta = full_feat_gen.create_aggregate_theta(
        best_model.penalized_theta,
        keep_col0=keep_col0,
        add_targets=add_targets,
    )
    return best_model

def load_logistic_model(file_name):
    with open(file_name, "r") as f:
        model = pickle.load(f)
    return model

def read_germline_file(fasta):
    """
    Read fasta file containing germlines

    @return dataframe with column "gene" for the name of the germline gene and
    "base" for the nucleotide content
    """

    with open(fasta) as fasta_file:
        genes = []
        bases = []
        for seq_record in SeqIO.parse(fasta_file, 'fasta'):
            genes.append(seq_record.id)
            bases.append(str(seq_record.seq))

    return pd.DataFrame({'base': bases}, index=genes)

def read_shmulate_val(shmulate_value):
    """ return the log so we can be sure we're comparing the same things!"""
    # the shazam csv puts an NA if we can never mutate to that target nucleotide
    # shazam csv puts a zero if there are not enough observations for that motif
    return -np.inf if shmulate_value == "NA" or shmulate_value == "0" or shmulate_value == "0.0" else np.log(float(shmulate_value))

def get_shazam_theta(mutability_file, substitution_file=None, wide_format=False):
    """
    Take shazam csv files and turn them into our theta vector

    @param feat_generator: feature generator for model
    @param mutability_file: csv of mutability fit from SHazaM
    @param substitution_file: csv of substitution fit from SHazaM
    """

    # Read in the results from the shmulate model-fitter
    # Read mutability matrix
    mut_motif_dict = dict()
    with open(mutability_file, "r") as model_file:
        if wide_format:
            csv_reader = csv.reader(model_file, delimiter=',')
            shazam_motif_list = csv_reader.next()[1:]
            shazam_mutabilities = csv_reader.next()[1:]
            for motif, motif_val in zip(shazam_motif_list, shazam_mutabilities):
                mut_motif_dict[motif.lower()] = motif_val
        else:
            csv_reader = csv.reader(model_file, delimiter=' ')
            header = csv_reader.next()
            for line in csv_reader:
                motif = line[0].lower()
                motif_val = line[1]
                mut_motif_dict[motif.lower()] = motif_val

    num_theta_cols = 1
    if substitution_file is not None:
        num_theta_cols = NUM_NUCLEOTIDES + 1
        # Read substitution matrix
        sub_motif_dict = dict()
        with open(substitution_file, "r") as model_file:
            if wide_format:
                csv_reader = csv.reader(model_file, delimiter=',')
            else:
                csv_reader = csv.reader(model_file, delimiter=' ')
            # Assume header is ACGT
            header = csv_reader.next()
            for i in range(NUM_NUCLEOTIDES):
                header[i + 1] = header[i + 1].lower()

            for line in csv_reader:
                motif = line[0].lower()
                mutate_to_prop = {}
                for i in range(NUM_NUCLEOTIDES):
                    mutate_to_prop[header[i + 1]] = line[i + 1]
                sub_motif_dict[motif] = mutate_to_prop


    # Shazam is always a 5mer
    feat_gen = MotifFeatureGenerator(motif_len=5)
    motif_list = feat_gen.motif_list
    # Reconstruct theta in the right order
    theta = np.zeros((feat_gen.feature_vec_len, num_theta_cols))
    for motif_idx, motif in enumerate(motif_list):
        theta[motif_idx, 0] = read_shmulate_val(mut_motif_dict[motif])
        if num_theta_cols > 1:
            for nuc in NUCLEOTIDES:
                theta[motif_idx, NUCLEOTIDE_DICT[nuc] + 1] = read_shmulate_val(sub_motif_dict[motif][nuc])

    return theta
