#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include "graph.h"
#include "test.h"

using namespace std;

bool differ_by_one_char(std::string s1, std::string s2){
  int num_diff = 0;
  for (int k = 0; k < 3; k++) {
    // printf("?? %s %s\n", s1[k], s2[k]);
    if (s1[k] != s2[k]) {
      num_diff++;
      if (num_diff > 1) {
        return false;
      }
    }
  }
  return num_diff == 1;
}

vector<string> get_motif_list(){
  int NUM_MOTIFS = 64;
  std::vector<std::string> motif_list(NUM_MOTIFS);

  char NUCLEOTIDES[] = "ATCG";
  int n = 0;
  for (int i=0; i < 4; i++) {
    for (int j=0; j < 4; j++) {
      for (int k=0; k < 4; k++) {
        motif_list[n] = std::string() + NUCLEOTIDES[i] + NUCLEOTIDES[j] + NUCLEOTIDES[k];
        printf("%s\n", motif_list[n].c_str());
        n++;
      }
    }
  }

  return motif_list;
}

void read_graph_vals(std::string file_name, double * graph_node_vals) {
  std::ifstream graph_file(file_name);
  std::string a;
  int i;
  double b;
  while (graph_file >> a >> i >> b) {
    // printf("%s %d %f\n", a.c_str(), i, b);
    graph_node_vals[i] = b;
  }
}

int solve_fused_lasso()
{
  std::string file_name = "test.txt";
  int NUM_MOTIFS = 64;

  double graph_node_vals[NUM_MOTIFS];
  read_graph_vals(file_name, graph_node_vals);

  double lambda = 0.001;

  double delta = 0.1;
  int numdeep = 14;
  int numlevel = (1 << numdeep) -1;
  printf("numlevel %d\n", numlevel);

  double lmin = 0.0;
  double lmax = 1.0;

  double * lev0 = (double *) malloc((numlevel+1)*sizeof(double));
  double * lev = (double *) malloc((numlevel+1)*sizeof(double));

  // construct these crazy level things.
  double dl=(lmax-lmin)/(double)numlevel;
  lev[0]=lev0[0]=lmin; //lev[0] never used
  for (int l=1;l<=numlevel;l++) {
    lev0[l]=dl+lev0[l-1];
    lev[l]=lev0[l-1]+dl/2.;
  }
  int idl = (1+numlevel) >> 1;

  vector<std::string>motif_list = get_motif_list();

  Graph::node_id * nodes = (Graph::node_id *) malloc((NUM_MOTIFS + 1)*sizeof(Graph::node_id));
  Graph *BKG = new Graph();

  // add nodes for each motif PLUS the last edge motif
  for (int i=0; i < NUM_MOTIFS + 1; i++) {
    nodes[i] = BKG->add_node();
    BKG->set_tweights(nodes[i], graph_node_vals[i], lev[idl]);
  }

  // add edges between motifs that differ by one character
  for (int i=0; i < NUM_MOTIFS; i++) {
    for (int j=0; j < NUM_MOTIFS; j++) {
      if (i == j) {
        continue;
      }
      bool is_diff = differ_by_one_char(motif_list.at(i), motif_list.at(j));
      // printf("%s, %s: %d\n", motif_list[i].c_str(), motif_list[j].c_str(), is_diff);
      if (is_diff) {
        // add edge
        BKG->add_edge(nodes[i],nodes[j],lambda,lambda);
      }
    }
  }

  // solve
  BKG->dyadicparametricTV(numdeep, idl*dl/2);
  //if (BKG->error()) { fprintf(stderr,"error in maxflow\n"); exit(0); }

  for (int k=0; k < NUM_MOTIFS + 1; k++)
    printf("%f\n", lev0[BKG->what_label(nodes[k])]);

  delete BKG;
  return 1;
}
