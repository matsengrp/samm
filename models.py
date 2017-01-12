class ObservedSequenceMutations:
    def __init__(self, start_seq, end_seq):
        assert(len(start_seq) == len(end_seq))

        self.start_seq = start_seq
        self.end_seq = end_seq

        # Make mutation position dictionary
        self.mutation_pos_dict = dict()
        for i in range(len(start_seq)):
            if start_seq[i] != end_seq[i]:
                self.mutation_pos_dict[i] = end_seq[i]

    def __str__(self):
        return "Seq %s, Mutations %s" % (
            self.start_seq,
            self.mutation_pos_dict,
        )

class ImputedSequenceMutations:
    def __init__(self, obs_seq_mutation, mutation_order):
        self.obs_seq_mutation = obs_seq_mutation
        self.mutation_order = mutation_order

    def __str__(self):
        return "Seq %s, Mutation Order %s" % (
            self.obs_seq_mutation.start_seq,
            self.mutation_order,
        )

class FullSequenceMutations:
    def __init__(self, obs_seq_mutation, mutations):
        """
        @param obs_seq_mutation: ObservedSequenceMutations
        @param mutations: a list of MutationPosTime
        """
        self.obs_seq_mutation = obs_seq_mutation
        self.mutations = mutations

    def __str__(self):
        return "%s => %s" % (
            self.obs_seq_mutation.start_seq,
            self.obs_seq_mutation.end_seq
        )

class MutationEvent:
    """
    Stores information on what happened during a mutation event
    """
    def __init__(self, time, pos, target_nucleotide):
        self.time = time
        self.pos = pos
        self.target_nucleotide = target_nucleotide

    def __str__(self):
        return "%d=%s" % (self.pos, self.target_nucleotide)
