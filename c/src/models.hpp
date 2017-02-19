#ifndef MODELS_HPP
#define MODELS_HPP

#include <string>
#include <vector>
#include <memory>

using namespace std;

typedef int Nuc; // 0 = A, 1 = C, 2 = G, 3 = T
// Nucleotide sequence
typedef struct {
  vector<Nuc> val;
} VectorNucleotide;

// Position indices by observed mutation order
typedef struct {
  vector<unsigned int> val;
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

    VectorNucleotide left_flank;
    VectorNucleotide right_flank;

    VectorFeature start_features;
    // positions are ordered just by position value
    // the order does not mean anything about mutation order!
    vector<unsigned int> mutated_postions;
    int num_pos;

    ObservedSample(
      VectorNucleotide s,
      VectorNucleotide e,
      VectorNucleotide l,
      VectorNucleotide r,
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
    ): nuc_vec(nucs), feature_vec(feats), theta_sum_option(t_sum_option) {};
};

class OrderedMutationSteps {
  public:
    int num_steps;
    VectorOrder order_vec;
    vector<shared_ptr<MutationStep> > mut_steps;

    // Initialize with the observed sample and the order under consideration
    OrderedMutationSteps(VectorOrder ord_v);

    OrderedMutationSteps(
      int num_s,
      VectorOrder order_v,
      vector<shared_ptr<MutationStep> > m_vec
    ): num_steps(num_s), order_vec(order_v), mut_steps(m_vec) {};

    // Initialize with the observed sample and the order under consideration
    shared_ptr<OrderedMutationSteps> copy(const shared_ptr<OrderedMutationSteps> &mut_order);

    // Update the mutation step with mut_step
    void set(
      int step_i,
      shared_ptr<MutationStep> mut_step
    );
};

#endif