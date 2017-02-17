#ifndef COMMON_H
#define COMMON_H

#include <string>
#include "models.hpp"

using namespace std;

namespace common {
  VectorNucleotide get_mutated_string(const VectorNucleotide &seq, int position, Nuc target_nuc);
}

#endif
