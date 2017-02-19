#include <random>
#include <iostream>

#include "common.hpp"

namespace common {
  VectorNucleotide GetMutatedNucleotideVector(const VectorNucleotide &nuc_vec, int position, Nuc target_nuc) {
    VectorNucleotide mutated_nucs(nuc_vec);
    mutated_nucs.val[position] = target_nuc;
    return mutated_nucs;
  }

  vector<double> GetRandomNormalRV(int num_rows, double mean, double sd) {
    // First create an instance of an engine.
    random_device rnd_device;
    // Specify the engine and distribution.
    mt19937 gen(rnd_device());
    normal_distribution<double> rnorm(mean, sd);

    vector<double> theta;
    for (int i = 0; i < num_rows; i++) {
      theta.push_back(rnorm(gen));
    }
    return theta;
  }
}
