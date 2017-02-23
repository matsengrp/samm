#include <random>
#include <iostream>

#include "common.hpp"

namespace common {
  const int NUM_NUCLEOTIDES = 4;

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

  VectorNucleotide GetRandomVectorNucleotide(int vec_len) {
    // First create an instance of an engine.
    random_device rnd_device;
    // Specify the engine and distribution.
    mt19937 gen(rnd_device());

    uniform_int_distribution<int> uniform_int_generator(0, NUM_NUCLEOTIDES - 1);

    VectorNucleotide nucs;
    for (int i = 0; i < vec_len; i++) {
      nucs.val.push_back(uniform_int_generator(gen));
    }

    return nucs;
  }
}
