#ifndef COMMON_HPP
#define COMMON_HPP

#include <string>
#include "models.hpp"

using namespace std;

namespace common {
  VectorNucleotide get_mutated_string(const VectorNucleotide &seq, int position, Nuc target_nuc);
}

#endif
