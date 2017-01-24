from survival_model import SurvivalModelSimulator

# TODO
simulator = SurvivalModelSimulator(true_theta, 0.1, feat_generator)
start_seqs = [
    "taagggaaaaaaaaaatgggggttt",
    "tttaaaaaagggggggggttgggttaaat",
]
for i in range(num_extra_seq):
    start_seqs.append(
        get_random_dna_seq(seq_length)
    )
