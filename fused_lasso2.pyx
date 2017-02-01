# distutils: sources=["test.cpp"]

cimport fused_lasso_fun

def happy():
    return fused_lasso_fun.solve_fused_lasso()
