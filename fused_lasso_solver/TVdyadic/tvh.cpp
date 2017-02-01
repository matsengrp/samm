#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <limits.h>

#include "graph.h"

extern "C" {
  void TV4(double *,int,int,double,int,double,double);
  void TV8(double *,int,int,double,int,double,double);
}
// TV-minimization 
// this can be linked with C code (tested only on linux) provided the
// flag -lstdc++ is given to the linker
// TV4 = with nearest neighbours interaction
// TV8 = with also next nearest
// TV16 = with 16 neighbours


void TV4(double *G, // solution is returned in the same array
	 int sx,int sy,// size of image
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
  double dval, dr,drd;

  Graph::node_id * nodes, no;

  if (numdeep <=0 || numdeep > 16 || sx <=2 || sy <= 2)
    { fprintf(stderr,"error: bad depth or bad size\n"); exit(0); }
  // levels ranges from 0 to numlevel (numlevel+1 values)
  numlevel = (1 << numdeep) -1;  // numdeep cannot be too big!
  // anyway limited by machine precision (of "double")

  lev0 = (double *) malloc((numlevel+1)*sizeof(double));
  lev = (double *) malloc((numlevel+1)*sizeof(double));
  nodes = (Graph::node_id *) malloc(sxy*sizeof(Graph::node_id));

  lambda *= 0.78539816339744830962 ; // pi/4
  // renormalization to ensure that the TV
  // is "as close as possible" to the isotropic TV [TV4 is anyway very far]

  if (lmax<=lmin) { lmax=1.; lmin=0.;  }
  dl=(lmax-lmin)/(double)numlevel;
  lev[0]=lev0[0]=lmin; //lev[0] never used
  for (l=1;l<=numlevel;l++) {
    lev0[l]=dl+lev0[l-1];
    lev[l]=lev0[l-1]+dl/2.;
  }

  dr=lambda;

  Graph *BKG = new Graph();
  idl= (1+numlevel) >> 1;

  for (i=0;i<sxy;i++) {
    no=nodes[i]=BKG->add_node();
    BKG->set_tweights(no,G[i],lev[idl]);
  }
  
  for (i=1;i<sx;i++) BKG->add_edge(nodes[i-1],nodes[i],dr,dr);
  for (j=1,k=sx;j<sy;j++) {
    BKG->add_edge(nodes[k-sx],nodes[k],dr,dr);
    for (i=1,k++;i<sx;i++,k++) {
      BKG->add_edge(nodes[k-1],nodes[k],dr,dr);
      BKG->add_edge(nodes[k-sx],nodes[k],dr,dr);
    }
  }
  BKG->dyadicparametricTV(numdeep,idl*dl/2);
  //if (BKG->error()) { fprintf(stderr,"error in maxflow\n"); exit(0); }
  
  for (k=0;k<sxy;k++) G[k]=lev0[BKG->what_label(nodes[k])];
  delete BKG;
  free(lev0); free(lev);
}

void TV8(double *G, // solution is returned in the same array
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
  double dval, dr,drd;

  Graph::node_id * nodes, no;

  if (numdeep <=0 || numdeep > 16 || sx <=2 || sy <= 2)
    { fprintf(stderr,"error: bad depth or bad size\n"); exit(0); }
  // levels ranges from 0 to numlevel (numlevel+1 values)
  numlevel = (1 << numdeep) -1; // numdeep cannot be too big!
  // anyway limited by machine precision (of "double")

  lev0 = (double *) malloc((numlevel+1)*sizeof(double));
  lev = (double *) malloc((numlevel+1)*sizeof(double));
  nodes = (Graph::node_id *) malloc(sxy*sizeof(Graph::node_id));

  lambda *= 0.39269908169872415481 ; // pi/8
  // pi/8: renormalization to ensure that the TV
  // is "as close as possible" to the isotropic TV


  if (lmax<=lmin) { lmax=1.; lmin=0.;  }
  dl=(lmax-lmin)/(double)numlevel;
  lev[0]=lev0[0]=lmin; //lev[0] is never used
  for (l=1;l<=numlevel;l++) {
    lev0[l]=dl+lev0[l-1];
    lev[l]=lev0[l-1]+dl/2.;
  }

  dr=lambda;
  drd=lambda*0.70710678118654752440;

  Graph *BKG = new Graph();
  idl= (1+numlevel) >> 1;

  for (i=0;i<sxy;i++) {
    no=nodes[i]=BKG->add_node();
    BKG->set_tweights(no,G[i],lev[idl]);
  }
  
  for (i=1;i<sx;i++) BKG->add_edge(nodes[i-1],nodes[i],dr,dr);
  for (j=1,k=sx;j<sy;j++) {
    BKG->add_edge(nodes[k-sx],nodes[k],dr,dr);
    for (i=1,k++;i<sx;i++,k++) {
      BKG->add_edge(nodes[k-1],nodes[k],dr,dr);
      BKG->add_edge(nodes[k-sx],nodes[k],dr,dr);
      BKG->add_edge(nodes[k-1],nodes[k-sx],drd,drd);
      BKG->add_edge(nodes[k-sx-1],nodes[k],drd,drd);
    }
  }
  BKG->dyadicparametricTV(numdeep,idl*dl/2);
  //if (BKG->error()) { fprintf(stderr,"error in maxflow\n"); exit(0); }
  
  for (k=0;k<sxy;k++) G[k]=lev0[BKG->what_label(nodes[k])];
  delete BKG;
  free(lev0); free(lev);
}
