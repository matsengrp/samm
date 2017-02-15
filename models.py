class ObservedSequenceMutations:
    def __init__(self, start_seq, end_seq, motif_len=1):
        assert(len(start_seq) == len(end_seq))

        start_index = 0
        end_index = len(start_seq)
        
        # A dictionary with key as position and value as target nucleotide
        self.mutation_pos_dict = dict()
        for i in range(len(start_seq)):
            if start_seq[i] != end_seq[i]:
                # ignore mutations happening close to edge
                if i < start_index + motif_len/2:
                    start_index = i + 1
                elif i > end_index - motif_len/2 - 1:
                    end_index = i
                else:
                    self.mutation_pos_dict[i - start_index] = end_seq[i]
        
        self.num_mutations = len(self.mutation_pos_dict.keys())
        self.start_seq = start_seq[start_index:end_index]
        self.end_seq = end_seq[start_index:end_index]
        self.seq_len = len(self.start_seq)
        assert(self.seq_len > 0)

    def __str__(self):
        return "Seq %s, Mutations %s" % (
            self.start_seq,
            self.mutation_pos_dict,
        )

class ImputedSequenceMutations:
    def __init__(self, obs_seq_mutation, mutation_order):
        """
        @param obs_seq_mutation: any object that needs to get augmented by a mutation order
                                (e.g. ObservedSequenceMutations or ObservedSequenceMutationsFeatures)
        @param mutation_order: a list of the positions in the order they mutated
        """
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
        @param mutations: an ordered list of MutationPosTime
        """
        self.obs_seq_mutation = obs_seq_mutation
        self.mutations = mutations

    def get_mutation_order(self):
        return [m.pos for m in self.mutations]

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
        return "%d=%s (%.2g)" % (self.pos, self.target_nucleotide, self.time)
