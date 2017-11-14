import csv
import numpy as np

from common import NUCLEOTIDES, NUM_NUCLEOTIDES, NUCLEOTIDE_DICT
from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator
from read_data import get_sequence_mutations_from_tree
from likelihood_evaluator import LogLikelihoodEvaluator

def _read_shazam_val(shazam_value):
    """ return the log so we can be sure we're comparing the same things!"""
    return -np.inf if shazam_value == "NA" else np.log(float(shazam_value))

def likelihood_of_tree_from_shazam(tree, mutability_file, substitution_file=None, feat_generator=None, num_jobs=1, scratch_dir='_output', num_samples=1000, burn_in=0):
    """
    Given an ETE tree and theta vector, compute the likelihood of that tree

    @param tree: an ETE tree
    @param mutability_file: csv of mutability fit from SHazaM
    @param substitution_file: csv of substitution fit from SHazaM; if empty assume all targets equiprobable
    @param feat_generator: feature generator for model; if empty then default to 5mer model
    @param num_jobs: how many jobs to run
    @param scratch_dir: where to put temporary output if running more than one job
    @param num_samples: number of chibs samples
    @param burn_in: number of burn-in iterations

    @return: log likelihood of a tree given a SHazaM fit
    """

    if feat_generator is None:
        # Default to S5F
        feat_generator = HierarchicalMotifFeatureGenerator(
            motif_lens=[5],
            left_motif_flank_len_list=[[2]],
        )

    theta_ref = shazam_to_theta(feat_generator, mutability_file, substitution_file)

    per_target_model = theta_ref.shape[1] == NUM_NUCLEOTIDES + 1

    obs_data = get_sequence_mutations_from_tree(
        tree,
        feat_generator.motif_len,
        feat_generator.max_left_motif_flank_len,
        feat_generator.max_right_motif_flank_len,
    )

    feat_generator.add_base_features_for_list(obs_data)
    log_like_evaluator = LogLikelihoodEvaluator(obs_data, feat_generator, num_jobs, scratch_dir)

    return log_like_evaluator.get_log_lik(
        theta_ref,
        num_samples=num_samples,
        burn_in=burn_in,
    )

def shazam_to_theta(feat_generator, mutability_file, substitution_file=None):
    """
    Take shazam csv files and turn them into our theta vector

    @param feat_generator: feature generator for model
    @param mutability_file: csv of mutability fit from SHazaM
    @param substitution_file: csv of substitution fit from SHazaM
    """

    # Read mutability matrix
    mut_motif_dict = dict()
    with open(mutability_file, "r") as model_file:
        csv_reader = csv.reader(model_file, delimiter=' ')
        for line in csv_reader:
            mut_motif_dict[line[0].lower()] = line[1]

    num_theta_cols = 1
    if substitution_file is not None:
        num_theta_cols = NUM_NUCLEOTIDES + 1
        # Read target matrix
        target_motif_dict = dict()
        with open(substitution_file, "r") as model_file:
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
                target_motif_dict[motif] = mutate_to_prop

    motif_list = feat_generator.motif_list
    # Reconstruct theta in the right order
    theta = np.zeros((feat_generator.feature_vec_len, num_theta_cols))
    for motif_idx, motif in enumerate(motif_list):
        theta[motif_idx, 0] = _read_shmulate_val(mut_motif_dict[motif])
        if num_theta_cols > 1:
            for nuc in NUCLEOTIDES:
                theta[motif_idx, NUCLEOTIDE_DICT[nuc] + 1] = _read_shmulate_val(target_motif_dict[motif][nuc])

    return theta

