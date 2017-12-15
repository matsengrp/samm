import unittest
import numpy as np

from sampler_collection import SamplerCollection
from mutation_order_gibbs import MutationOrderGibbsSampler
from hier_motif_feature_generator import HierarchicalMotifFeatureGenerator
from survival_model_simulator import SurvivalModelSimulatorPositionDependent
from models import ObservedSequenceMutations
from common import get_possible_motifs_to_targets, get_random_dna_seq, NUM_NUCLEOTIDES

class Position_Simulation_TestCase(unittest.TestCase):
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

    def _generate_data(self, n_naive=100, random_gene_len=40):
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

        # the first half of the gene mutates less, second half mutates more
        pos_risk = [-1] * (random_gene_len / 2) + [1] * (random_gene_len / 2)

        target_shape = (self.feat_gen.feature_vec_len, NUM_NUCLEOTIDES)
        probability_matrix = np.ones(target_shape) * 1.0/3
        possible_motifs_mask = get_possible_motifs_to_targets(self.feat_gen.motif_list, target_shape, 
                                                              self.feat_gen.mutating_pos_list)
        probability_matrix[~possible_motifs_mask] = 0.
        simulator = SurvivalModelSimulatorPositionDependent(theta, probability_matrix, self.feat_gen, 
                                                            lambda0=0.1, pos_risk=pos_risk)

        n_mut_left = []
        n_mut_right = []
        for _ in range(n_naive):
            germline_sequence = get_random_dna_seq(random_gene_len)
            sample = simulator.simulate(
                start_seq=germline_sequence.lower(),
                percent_mutated=np.random.uniform(low=.05, high=.20),
            )
            end_seq = sample.left_flank + sample.end_seq + sample.right_flank
            # count the number of mutations on the left and right halves of the sequence
            mutation_indicator = [germline_sequence[i] != end_seq[i] for i in range(random_gene_len)]
            n_mut_left.append(np.sum(mutation_indicator[:(random_gene_len / 2)]))
            n_mut_right.append(np.sum(mutation_indicator[(random_gene_len / 2):]))

        return n_mut_left, n_mut_right

    def test_position_dependence(self):
        n_mut_left, n_mut_right = self._generate_data()
        # do a t test for n_mut_right > n_mut_left
        diffs = [right - left for (right, left) in zip(n_mut_right, n_mut_left)]
        sd = np.std(diffs)
        n = len(diffs)
        t_statistic = np.mean(diffs) / (sd / np.sqrt(n - 1))
        self.assertTrue(t_statistic >= 2)
