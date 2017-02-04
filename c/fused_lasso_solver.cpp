#include <stdio.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include "graph.h"
#include "fused_lasso_solver.h"

using namespace std;

void solve_fused_lasso(
  double *array_to_fill,
  double *feature_vals,
  int vector_size,
  int *features_to_fuse1,
  int *features_to_fuse2,
  int num_features_to_fuse,
  double penalty_param,
  double err
) {
  // This is going to find the minimizer of
  // 0.5 * || theta - g ||^2_2 + lambda * fused lasso penalty

  // Now construct the graph
  Graph::node_id *nodes = new Graph::node_id[vector_size];
  Graph *BKG = new Graph();

  for (int i = 0; i < vector_size; i++) {
    nodes[i] = BKG->add_node();
    BKG->set_tweights(
      nodes[i],
      feature_vals[i],
      0
    );
  }

  // add edges between motifs that differ by one character
  for (int i = 0; i < num_features_to_fuse; i++) {
    int node1 = features_to_fuse1[i];
    int node2 = features_to_fuse2[i];
    BKG->add_edge(nodes[node1], nodes[node2], penalty_param, penalty_param);
  }

  // solve max flow problem
  BKG->dyadicparametricTV(err);
  
  // Output the fitted values
  for (int i = 0; i < vector_size; i++) {
    array_to_fill[i] = BKG->what_value(nodes[i]);
  }

  delete BKG;
  delete[] nodes;
}
