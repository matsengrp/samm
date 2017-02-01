# distutils: language=c++
# distutils: sources=["c/fused_lasso_solver.cpp"]

import cython
import ctypes
import numpy as np
cimport numpy as np

cimport fused_lasso_fun


@cython.boundscheck(False)
@cython.wraparound(False)
def solve_fused_lasso(
    np.ndarray[double, ndim=1, mode="c"] input not None,
    np.ndarray[double, ndim=1, mode="c"] feature_vals not None,
    np.ndarray[int, ndim=1, mode="c"] features_to_fuse1 not None,
    np.ndarray[int, ndim=1, mode="c"] features_to_fuse2 not None,
    double penalty_param,
    int numdeep = 16,
  ):
    fused_lasso_fun.solve_fused_lasso(
      &input[0],
      &feature_vals[0],
      feature_vals.size,
      &features_to_fuse1[0],
      &features_to_fuse2[0],
      features_to_fuse2.size,
      penalty_param,
      np.min(feature_vals),
      np.max(feature_vals),
      numdeep,
    )
