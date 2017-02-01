This is an implementation of Dorit Hochbaum's algorithm:
     D. S. Hochbaum: An efficient algorithm for image segmentation,
     Markov random fields and related problems. J. ACM, 48(4):686--701,
     2001.

following:
     A. Chambolle and J. Darbon: On total variation minimization and
     surface evolution using parametric maximum flows, preprint (2008)

For more information on these files get the README file in
the original package
maxflow-v2.2.src.tar.gz (directory adjacency_list)
available at
http://www.adastral.ucl.ac.uk/~vladkolm/software.html
(Vladimir Kolmogorov's home page)

Our small code has been tested only on linux systems.
To compile:

cc -O -c *.cpp
ar rus libTVH.a *.o

to link with your C code
cc <your flags, files> -L<path_to_libTVH.a> -lTVH -lstdc++

Antonin Chambolle and Jérôme Darbon
