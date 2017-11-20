import csv
import numpy as np

from common import NUCLEOTIDES, NUM_NUCLEOTIDES, NUCLEOTIDE_DICT
from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator
from read_data import get_sequence_mutations_from_tree, get_shazam_theta
from likelihood_evaluator import LogLikelihoodEvaluator

def likelihood_of_tree_from_shazam(tree, mutability_file, substitution_file=None, num_jobs=1, scratch_dir='_output', num_samples=1000, burn_in=0, num_tries=5):
    """
    Given an ETE tree and theta vector, compute the likelihood of that tree

    @param tree: an ETE tree
    @param mutability_file: csv of mutability fit from SHazaM
    @param substitution_file: csv of substitution fit from SHazaM; if empty assume all targets equiprobable
    @param num_jobs: how many jobs to run
    @param scratch_dir: where to put temporary output if running more than one job
    @param num_samples: number of chibs samples
    @param burn_in: number of burn-in iterations
    @param num_tries: number of tries for Chibs sampler

    @return: log likelihood of a tree given a SHazaM fit
    """

    # Default for SHazaM is S5F
    feat_generator = HierarchicalMotifFeatureGenerator(
        motif_lens=[5],
        left_motif_flank_len_list=[[2]],
    )

    theta_ref = get_shazam_theta(feat_generator.max_motif_len, mutability_file, substitution_file)

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
        num_tries=num_tries,
    )

