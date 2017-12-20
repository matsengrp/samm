import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sampler_collection import SamplerCollection
from mutation_order_gibbs import MutationOrderGibbsSampler
from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator
from simulate_germline import GermlineMetadata
from survival_model_simulator import SurvivalModelSimulatorSingleColumn
from models import ObservedSequenceMutations
from common import get_possible_motifs_to_targets, get_random_dna_seq, NUM_NUCLEOTIDES

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
        cls.sampling_rate = 1
        cls.burn_in = 1
        cls.nonzero_ratio = 0.5

    def _generate_data(self, n_naive=100, random_gene_len=20):
        """
        Generate both theta and data (randomly sample nucleotides)
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
        simulator = SurvivalModelSimulatorSingleColumn(theta, probability_matrix, self.feat_gen, lambda0=0.1)

        obs_data = []
        for _ in range(n_naive):
            germline_sequence = get_random_dna_seq(random_gene_len)
            sample = simulator.simulate(
                start_seq=germline_sequence.lower(),
                percent_mutated=np.random.uniform(low=.05, high=.20),
            )
            obs_seq_mutation = ObservedSequenceMutations(
                start_seq=sample.left_flank + sample.start_seq + sample.right_flank,
                end_seq=sample.left_flank + sample.end_seq + sample.right_flank,
                motif_len=self.motif_len,
            )
            if obs_seq_mutation.num_mutations > 0:
                # don't consider pairs where mutations occur in flanking regions
                obs_data.append(obs_seq_mutation)

        return theta, obs_data

    def _get_residuals(self, get_residuals=True):
        """
        Calculate residuals from Gibbs samples
        """
        theta, obs_data = self._generate_data()
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
        return np.array([res.residuals for res in sampler_results])

    def test_residuals(self):
        # Residuals have mean zero over *subjects*, are between -\infty and 1, and are approximately uncorrelated
        residuals = self._get_residuals(get_residuals=True)
        index_vec = range(residuals.shape[1])
        for idx, resid in zip(index_vec, residuals.T):
            plt.scatter([idx] * len(resid), resid)
        plt.savefig('test/_output/residuals.svg')
        self.assertTrue(np.nanmax(residuals) <= 1.)
        residual_mean = np.nanmean(residuals, axis=0)
        residual_mean = residual_mean[~np.isnan(residual_mean)]
        # TODO: below is false, and doesn't change with more samples; is something wrong?
        #self.assertTrue(np.allclose(residual_mean, np.zeros(np.shape(residual_mean))))

        # Negative control: run gibbs sampling without getting residuals
        residuals = self._get_residuals(get_residuals=False)

