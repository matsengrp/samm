# distutils: language=c++
from libcpp.string cimport string

cdef extern from "c/fused_lasso_solver.h":
    void solve_fused_lasso(
      double *array,
      double *feature_vals,
      int vector_size,
      int *feature_to_fuse1,
      int *feature_to_fuse2,
      int num_features_to_fuse,
      double penalty_param,
      double lmin,
      double lmax,
      int numdeep
    )
