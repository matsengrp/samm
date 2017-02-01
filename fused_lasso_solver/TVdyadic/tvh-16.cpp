#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <limits.h>

#include "graph.h"

extern "C" {
  void TV16(double *,int,int,double,int,double,double);
}
// TV-minimization 
// this can be linked with C code (tested only on linux) provided the
// flag -lstdc++ is given to the linker
// TV4 = with nearest neighbours interaction
// TV8 = with also next nearest
// TV16 = with 16 neighbours

void TV16(double *G, // solution is returned in the same array
	  int sx,int sy, // size of image
	  double lambda, // weight on the total variation
	  int numdeep, double lmin, double lmax)
// numdeep = depth of the dyadic search: 2^numdeep levels are actually
// computed (hence the precision is (lmax-lmin)*2^(-numdeep))
// the solution is truncated to lmin/lmax: for an optimal result with
// no truncation they must be set to the actual min/max of the original G.

{

  int i,j,k,l,idl,sxy=sx*sy;
  double r,dl,*lev,*lev0;
  unsigned int numlevel, ldyad;
  Graph::captype dr,drd,dr12;

  Graph::node_id * nodes, no;

  if (numdeep <=0 || numdeep > 16 || sx <=2 || sy <= 2)
    { fprintf(stderr,"error: bad depth or bad size\n"); exit(0); }
  // levels ranges from 0 to numlevel (numlevel+1 values)
  numlevel = (1 << numdeep) -1; // numdeep cannot be too big!
  // anyway limited by machine precision (of "double")

  lev0 = (double *) malloc((numlevel+1)*sizeof(double));
  lev = (double *) malloc((numlevel+1)*sizeof(double));
  nodes = (Graph::node_id *) malloc(sxy*sizeof(Graph::node_id));

  lambda *= 0.39269908169872415481/2.; ;
  // pi/16: renormalization to ensure that the TV
  // is "closer" to the isotropic TV

  if (lmax<=lmin) { lmax=1.; lmin=0.;  }
  dl=(lmax-lmin)/(double)numlevel;
  lev[0]=lev0[0]=lmin; //lev[0] never used
  for (l=1;l<=numlevel;l++) {
    lev0[l]=dl+lev0[l-1];
    lev[l]=lev0[l-1]+dl/2.;
  }

  dr=lambda;
  drd=lambda*0.70710678118654752440;
  dr12=lambda*.447213595499957939281834;

  Graph *BKG = new Graph();
  idl= (1+numlevel) >> 1;

  for (i=0;i<sxy;i++) {
    no=nodes[i]=BKG->add_node();
    BKG->set_tweights(no,G[i],lev[idl]);
  }
  k=1;
  for (i=1;i<sx;i++,k++) BKG->add_edge(nodes[k-1],nodes[k],dr,dr);
  BKG->add_edge(nodes[k-sx],nodes[k],dr,dr);
  k++;
  BKG->add_edge(nodes[k-1],nodes[k],dr,dr);
  BKG->add_edge(nodes[k-sx],nodes[k],dr,dr);
  BKG->add_edge(nodes[k-1],nodes[k-sx],drd,drd);
  BKG->add_edge(nodes[k],nodes[k-1-sx],drd,drd);
  for (i=2,k++;i<sx;i++,k++) {
    BKG->add_edge(nodes[k-1],nodes[k],dr,dr);
    BKG->add_edge(nodes[k-sx],nodes[k],dr,dr);
    BKG->add_edge(nodes[k-1],nodes[k-sx],drd,drd);
    BKG->add_edge(nodes[k],nodes[k-1-sx],drd,drd);
    BKG->add_edge(nodes[k],nodes[k-2-sx],dr12,dr12);
    BKG->add_edge(nodes[k-sx],nodes[k-2],dr12,dr12);
  }
  for (j=2;j<sy;j++) {
    BKG->add_edge(nodes[k-sx],nodes[k],dr,dr);
    k++;
    BKG->add_edge(nodes[k-1],nodes[k],dr,dr);
    BKG->add_edge(nodes[k-sx],nodes[k],dr,dr);
    BKG->add_edge(nodes[k-1],nodes[k-sx],drd,drd);
    BKG->add_edge(nodes[k],nodes[k-1-sx],drd,drd);
    BKG->add_edge(nodes[k],nodes[k-2*sx-1],dr12,dr12);
    BKG->add_edge(nodes[k-2*sx],nodes[k-1],dr12,dr12);
    for (i=2,k++;i<sx;i++,k++) {
      BKG->add_edge(nodes[k-1],nodes[k],dr,dr);
      BKG->add_edge(nodes[k-sx],nodes[k],dr,dr);
      BKG->add_edge(nodes[k-1],nodes[k-sx],drd,drd);
      BKG->add_edge(nodes[k],nodes[k-1-sx],drd,drd);      
      BKG->add_edge(nodes[k],nodes[k-2*sx-1],dr12,dr12);
      BKG->add_edge(nodes[k-2*sx],nodes[k-1],dr12,dr12);
      BKG->add_edge(nodes[k],nodes[k-2-sx],dr12,dr12);
      BKG->add_edge(nodes[k-sx],nodes[k-2],dr12,dr12);
    }
  }
  BKG->dyadicparametricTV(numdeep,idl*dl/2);
  //if (BKG->error()) { fprintf(stderr,"error in maxflow\n"); exit(0); }
  
  for (k=0;k<sxy;k++) G[k]=lev0[BKG->what_label(nodes[k])];
  delete BKG;
  free(lev0); free(lev);
  free(nodes);
}
