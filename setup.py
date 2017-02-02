###
# To compile this stuff, call
# python setup.py build_ext --inplace
###

import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

exts = cythonize([Extension(
        "solver_wrappers",
        sources=["solver_wrappers.pyx", "c/fused_lasso_solver.cpp"],
        include_dirs=["c/TVexact", np.get_include()],
        libraries=["TVH", "stdc++"],
        library_dirs=["c/TVexact"]
    )],
    language="c++"
)

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = exts,
)
