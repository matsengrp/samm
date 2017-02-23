#ifndef COMMON_HPP
#define COMMON_HPP

#include <iostream>
#include <string>
#include "models.hpp"

using namespace std;

namespace common {
  VectorNucleotide GetMutatedNucleotideVector(const VectorNucleotide &nuc_vec, int position, Nuc target_nuc);

  vector<double> GetRandomNormalRV(int motif_len, double mean=0, double sd=1);

  VectorNucleotide GetRandomVectorNucleotide(int vec_len);

  template <typename T>
  void PrintVector(vector<T> v) {
    for (auto i: v) {
      cout << i << " ";
    }
    cout << "\n";
  };
}

#endif
