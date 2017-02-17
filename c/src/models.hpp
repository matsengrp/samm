#ifndef MODELS_HPP
#define MODELS_HPP

#include <string>
#include <vector>

using namespace std;

typedef int Nuc; // 0 = A, 1 = C, 2 = G, 3 = T
// Nucleotide sequence
typedef struct {
  vector<Nuc> val;
} VectorNucleotide;

// Position indices by observed mutation order
typedef struct {
  vector<int> val;
} VectorOrder;

// Indices of the feature at each position
// Positions at risk of mutating have a nonnegative feature index
// Positions no longer in the risk group have a MUTATED=-1 feature index
typedef struct {
  vector<int> val;
} VectorFeature;
// Theta * psi for every position in sequence
// Check if position in VectorThetaSums is valid by checking if position is in risk group
// So check VectorFeature
typedef struct {
  vector<double> val;
} VectorThetaSums;

class ObservedSample {
  public:
    VectorNucleotide start_nucs;
    VectorNucleotide end_nucs;
    VectorFeature start_features;
    int num_pos;

    ObservedSample(
      VectorNucleotide s,
      VectorNucleotide e,
      VectorFeature f
    );
};

class MutationStep {
  public:
    VectorNucleotide nuc_vec;
    VectorFeature feature_vec;
    // A container that might contain a VectorThetaSums object
    // CHeck if the first elem is True before reading the second value
    pair<bool, VectorThetaSums> theta_sum_option;

    MutationStep(
      VectorNucleotide nucs,
      VectorFeature feats,
      pair<bool, VectorThetaSums> t_sum_option
    ): nuc_vec{nucs}, feature_vec{feats}, theta_sum_option{t_sum_option} {};
};

class OrderedMutationSteps {
  public:
    int num_steps;
    VectorOrder order_vec;
    vector<shared_ptr<MutationStep>> mut_steps;

    // Initialize with the observed sample and the order under consideration
    OrderedMutationSteps(VectorOrder ord_v);

    // Update the mutation step with mut_step
    void set(
      int step_i,
      shared_ptr<MutationStep> mut_step
    );
};

#endif
