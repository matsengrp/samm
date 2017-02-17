#include "common.hpp"

namespace common {
  VectorNucleotide get_mutated_nucleotide_vector(const VectorNucleotide &nuc_vec, int position, Nuc target_nuc) {
    VectorNucleotide mutated_nucs(nuc_vec);
    mutated_nucs.val[position] = target_nuc;
    return mutated_nucs;
  }
}
