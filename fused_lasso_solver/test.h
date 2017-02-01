#include <string>
using namespace std;

#ifndef FUSED_LASSO_H
#define FUSED_LASSO_H

int solve_fused_lasso(double lmin=-5.0, double lmax=5.0, int numdeep=16, const char *infile_name = "in.txt", const char *outfile_name="out.txt");

#endif
