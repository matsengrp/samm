#include "common.hpp"

namespace common {
  NucSeq mutate_string(const NucSeq &seq, int position, Nuc target_nuc) {
    NucSeq mutated_seq(seq);
    mutated_seq.val[position] = target_nuc;
    return mutated_seq;
  }
}
