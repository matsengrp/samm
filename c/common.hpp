#ifndef COMMON_H
#define COMMON_H

#include <string>
#include "models.hpp"

using namespace std;

namespace common {
  NucSeq mutate_string(const NucSeq &seq, int position, Nuc target_nuc);
}

#endif
