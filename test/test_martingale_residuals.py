import unittest
import numpy as np

import matplotlib
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sampler_collection import SamplerCollection
from mutation_order_gibbs import MutationOrderGibbsSampler
from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator
from simulate_germline import GermlineMetadata
from survival_model_simulator import SurvivalModelSimulatorSingleColumn, SurvivalModelSimulatorPositionDependent
from models import ObservedSequenceMutations
from common import get_possible_motifs_to_targets, get_random_dna_seq, NUM_NUCLEOTIDES, process_degenerates_and_impute_nucleotides
from plot_helpers import plot_martingale_residuals_on_axis

POSITION_BIAS = 3

class Residuals_TestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Set up state
        """
        np.random.seed(1)
        cls.motif_len = 3
        cls.mut_pos_list = [[1]]
        cls.feat_gen = HierarchicalMotifFeatureGenerator(
            motif_lens=[cls.motif_len],
            left_motif_flank_len_list=cls.mut_pos_list,
        )
        cls.num_jobs = 1
        cls.scratch_dir = 'test/_output/'
        cls.num_e_samples = 4
        cls.sampling_rate = 2
        cls.burn_in = 1
        cls.nonzero_ratio = 0.5

    def _generate_data(self, n_naive=1000, max_gene_len=20, min_gene_len=None, position_bias=False):
        """
        Generate both theta and data (randomly sample nucleotides)

        @param n_naive: number of naive sequences to simulate
        @param max_gene_len: maximum length of germline sequence to simulate
        @param min_gene_len: minum length of germline sequence to simulate; if None then just simulate max_gene_len length sequences
        @param position_bias: if True, simulate data where the first half of each sequence mutates less frequently than latter half

        @return randomly generated theta numpy array
        @return list of ObservedSequenceMutations of simulated data
        """
        theta_shape = (self.feat_gen.feature_vec_len, 1)
        theta = np.random.normal(size=theta_shape)

        # Zero parts of the first theta column
        indices_to_zero = np.random.choice(
            np.arange(theta.size),
            size=int((1 - self.nonzero_ratio) * theta.size),
            replace=False,
        )
        theta[indices_to_zero] = 0.
        theta -= np.median(theta)

        target_shape = (self.feat_gen.feature_vec_len, NUM_NUCLEOTIDES)
        probability_matrix = np.ones(target_shape) * 1.0/3
        possible_motifs_mask = get_possible_motifs_to_targets(self.feat_gen.motif_list, target_shape, self.feat_gen.mutating_pos_list)
        probability_matrix[~possible_motifs_mask] = 0.

        # Default simulator
        simulator = SurvivalModelSimulatorSingleColumn(theta, probability_matrix, self.feat_gen, lambda0=0.1)

        obs_data = []
        for _ in range(n_naive):
            if min_gene_len is None:
                curr_gene_len = max_gene_len
            else:
                curr_gene_len = np.random.randint(min_gene_len, max_gene_len)
            germline_sequence = get_random_dna_seq(curr_gene_len)
            if position_bias:
                # the first half of the gene mutates less, second half mutates more
                pos_risk = [-POSITION_BIAS] * (curr_gene_len / 2) + [POSITION_BIAS] * (curr_gene_len - curr_gene_len / 2)
                simulator = SurvivalModelSimulatorPositionDependent(theta, probability_matrix, self.feat_gen,
                                                                    lambda0=0.1, pos_risk=pos_risk)
            sample = simulator.simulate(
                start_seq=germline_sequence.lower(),
                percent_mutated=np.random.uniform(low=.05, high=.20),
            )
            # Make a quarter of sequences have interior degenerate nucleotides
            raw_start_seq = sample.left_flank + sample.start_seq + sample.right_flank
            raw_end_seq = sample.left_flank + sample.end_seq + sample.right_flank
            if np.random.uniform() < .25:
                raw_start_seq = raw_start_seq[:curr_gene_len/2] + 'n' * self.motif_len * 2 + raw_start_seq[curr_gene_len/2:]
                raw_end_seq = raw_end_seq[:curr_gene_len/2] + 'n' * self.motif_len * 2 + raw_end_seq[curr_gene_len/2:]

            start_seq, end_seq, collapse_list = process_degenerates_and_impute_nucleotides(
                raw_start_seq,
                raw_end_seq,
                self.motif_len,
            )

            obs_seq_mutation = ObservedSequenceMutations(
                start_seq=start_seq,
                end_seq=end_seq,
                motif_len=self.motif_len,
                collapse_list=collapse_list,
            )
            if obs_seq_mutation.num_mutations > 0:
                # don't consider pairs where mutations occur in flanking regions
                obs_data.append(obs_seq_mutation)

        return theta, obs_data

    def _get_residuals(self, get_residuals=True, max_gene_len=20, min_gene_len=None, position_bias=False):
        """
        Calculate residuals from Gibbs samples

        @param get_residuals: if True, calculate residuals; only included for negative control
        @param max_gene_len: maximum length of germline sequence to simulate
        @param min_gene_len: minum length of germline sequence to simulate; if None then just simulate max_gene_len length sequences
        @param position_bias: if True, simulate data where the first half of each sequence mutates less frequently than latter half

        @return list of GibbsSamplerResults
        """
        theta, obs_data = self._generate_data(min_gene_len=min_gene_len, max_gene_len=max_gene_len, position_bias=position_bias)
        self.feat_gen.add_base_features_for_list(obs_data)
        sampler_collection = SamplerCollection(
            obs_data,
            theta,
            MutationOrderGibbsSampler,
            self.feat_gen,
            self.num_jobs,
            self.scratch_dir,
            get_residuals=get_residuals,
        )
        init_orders = [
            np.random.permutation(obs_seq.mutation_pos_dict.keys()).tolist()
            for obs_seq in obs_data
        ]
        sampler_results = sampler_collection.get_samples(
            init_orders,
            self.num_e_samples,
            self.burn_in,
            sampling_rate=self.sampling_rate,
        )
        return sampler_results

    def test_residuals(self):
        # Residuals have mean zero over subjects and are between -\infty and 1
        # Correctly specified model
        sampler_results = self._get_residuals(min_gene_len=15, max_gene_len=25, get_residuals=True)
        fig, axs = plt.subplots(ncols=1, nrows=1)
        residuals = plot_martingale_residuals_on_axis(
            [res.residuals for res in sampler_results],
            axs,
            trim_proportion=.5,
            plot_average=True,
            title='Residuals vs. Position (correctly specified model)',
            xlabel='residual',
        )
        plt.savefig('test/_output/residuals.svg')
        plt.clf()
        self.assertTrue(np.nanmax(residuals) <= 1.)
        self.assertTrue(np.isclose(np.nanmean(residuals), 0.))

        # Positional bias
        sampler_results = self._get_residuals(get_residuals=True, position_bias=True)
        fig, axs = plt.subplots(ncols=1, nrows=1)
        residuals = plot_martingale_residuals_on_axis(
            [res.residuals for res in sampler_results],
            axs,
            plot_average=True,
            title='Residuals vs. Position (position-biased model)',
            xlabel='residual',
        )
        plt.savefig('test/_output/residuals_position_bias.svg')
        plt.clf()
        self.assertTrue(np.nanmax(residuals) <= 1.)
        self.assertTrue(np.isclose(np.nanmean(residuals), 0.))

        # Negative control: run gibbs sampling without getting residuals
        residuals = self._get_residuals(get_residuals=False)

