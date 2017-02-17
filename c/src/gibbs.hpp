#ifndef GIBBS_H
#define GIBBS_H

#include <string>
using namespace std;

// Get gibbs samples
void run_gibbs(
  double *theta,
  int submotif_len,
  string start_seq,
  string end_seq,
  int *init_order, // initial order
  int burn_in, // number of burn in iters
  int num_samples // number of samples
);

#endif
