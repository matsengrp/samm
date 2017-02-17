#include "common.hpp"

namespace common {
  VectorNucleotide get_mutated_string(const VectorNucleotide &seq, int position, Nuc target_nuc) {
    VectorNucleotide mutated_seq(seq);
    mutated_seq[position] = target_nuc;
    return mutated_seq;
  }
}
