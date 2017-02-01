/* graphtv.cpp */
/*
    Copyright 2001 Vladimir Kolmogorov (vnk@cs.cornell.edu), Yuri Boykov (yuri@csd.uwo.ca).

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    ------------------------------
    Copyright 2007 Antonin Chambolle (antonin.chambolle@polytechnique.fr),
                   Jérôme Darbon (jerome@math.ucla.edu)

    This is a modified version of the Graph::maxflow() program of Kolmogorov
    and Boykov (available at
    http://www.adastral.ucl.ac.uk/~vladkolm/software/maxflow-v2.2.src.tar.gz)
    by Antonin Chambolle and Jérôme Darbon, to implement a parametric/dyadic
    graphcut for minimizing the total variation, as suggested by
    Dorit Hochbaum [1] (see also [2,3]). It is (experimentally) faster than
    a preflow implementation (described in [1]). Please cite [4] if you
    use this code.

    This must be compiled together with the files from
    maxflow-v2.2/adjacency_list distributed by Kolmogorov
    and Boykov. Be careful, graph.h is slightly modified -- the following
    lines are added or modified:
    typedef double captype;
    typedef double flowtype;
    int what_label(node_id i);
    void dyadicparametricTV(int, captype);
    unsigned short label; // for Hochbaum's parametric TV

    [1] D. S. Hochbaum: An efficient algorithm for image segmentation,
        Markov random fields and related problems. J. ACM, 48(4):686--701,
	2001.
    [2] J. Darbon and M. Sigelle: Image restoration with discrete
        constrained Total Variation part I: Fast and exact optimization.
	Journal of Mathematical Imaging and Vision, 26(3):261--276, 2006.
    [3] A. Chambolle: Total variation minimization and a class of binary
        MRF models. in EMMCVPR'05, LNCS #3757, 136--152, 2005.
    [4] A. Chambolle and J. Darbon: On total variation minimization and
        surface evolution using parametric maximum flows, preprint (2008)
*/


#include <stdio.h>
#include "maxflow.cpp"

/***********************************************************************/

void Graph::dyadicparametricTV(int numlevel,captype delta)
  // paramétrique (Hochbaum)
{
	node *i, *j, *current_node = NULL;
	arc *a;
	nodeptr *np, *np_next;
	int l, deltalabel;

      for (l=numlevel-1;l>=0;l--) {
	maxflow_init();
	deltalabel=1<<l;

	nodeptr_block = new DBlock<nodeptr>(NODEPTR_BLOCK_SIZE, error_function);
	while ( 1 )
	{
		if (i=current_node)
		{
			i -> next = NULL; /* remove active flag */
			if (!i->parent) i = NULL;
		}
		if (!i)
		{
			if (!(i = next_active())) break;
		}

		/* growth */
		if (!i->is_sink)
		{
			/* grow source tree */
			for (a=i->first; a; a=a->next)
			if (a->r_cap)
			{
				j = a -> head;
				if (!j->parent)
				{
					j -> is_sink = 0;
					j -> parent = a -> sister;
					j -> TS = i -> TS;
					j -> DIST = i -> DIST + 1;
					set_active(j);
				}
				else if (j->is_sink) break;
				else if (j->TS <= i->TS &&
				         j->DIST > i->DIST)
				{
					/* heuristic - trying to make the distance from j to the source shorter */
					j -> parent = a -> sister;
					j -> TS = i -> TS;
					j -> DIST = i -> DIST + 1;
				}
			}
		}
		else
		{
			/* grow sink tree */
			for (a=i->first; a; a=a->next)
			if (a->sister->r_cap)
			{
				j = a -> head;
				if (!j->parent)
				{
					j -> is_sink = 1;
					j -> parent = a -> sister;
					j -> TS = i -> TS;
					j -> DIST = i -> DIST + 1;
					set_active(j);
				}
				else if (!j->is_sink) { a = a -> sister; break; }
				else if (j->TS <= i->TS &&
				         j->DIST > i->DIST)
				{
					/* heuristic - trying to make the distance from j to the sink shorter */
					j -> parent = a -> sister;
					j -> TS = i -> TS;
					j -> DIST = i -> DIST + 1;
				}
			}
		}

		TIME ++;

		if (a)
		{
			i -> next = i; /* set active flag */
			current_node = i;

			/* augmentation */
			augment(a);
			/* augmentation end */

			/* adoption */
			while (np=orphan_first)
			{
				np_next = np -> next;
				np -> next = NULL;

				while (np=orphan_first)
				{
					orphan_first = np -> next;
					i = np -> ptr;
					nodeptr_block -> Delete(np);
					if (!orphan_first) orphan_last = NULL;
					if (i->is_sink) process_sink_orphan(i);
					else            process_source_orphan(i);
				}

				orphan_first = np_next;
			}
			/* adoption end */
		}
		else current_node = NULL;
	}

	delete nodeptr_block;
	if (l)
	  for (i=node_block->ScanFirst(); i; i=node_block->ScanNext()) {
	    if (what_segment(i)==SOURCE) {
	      i->label += deltalabel;
	      i->tr_cap -= delta;
	      // contracter les arcs: [HOW?]
	    } else {
	      i->tr_cap += delta;
	      for (a=i->first; a; a=a->next)
	      	if (what_segment(a->head)==SOURCE) { a->r_cap=0;}
	    }
	  }
	else
	  for (i=node_block->ScanFirst(); i; i=node_block->ScanNext())
	    if (what_segment(i)==SOURCE) i->label += deltalabel;
	delta /= 2;
      }
}

/***********************************************************************/

int Graph::what_label(node_id i) { return (int) (((node*) i)->label) ; }
