/*
 * Copyright 1999, 2005 Brown University, Providence, RI.
 * 
 *                         All Rights Reserved
 * 
 * Permission to use, copy, modify, and distribute this software and its
 * documentation for any purpose other than its incorporation into a
 * commercial product is hereby granted without fee, provided that the
 * above copyright notice appear in all copies and that both that
 * copyright notice and this permission notice appear in supporting
 * documentation, and that the name of Brown University not be used in
 * advertising or publicity pertaining to distribution of the software
 * without specific, written prior permission.
 * 
 * BROWN UNIVERSITY DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
 * INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL BROWN UNIVERSITY BE LIABLE FOR
 * ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#include "FeatTreeIter.h"

void
FeatTreeIter::
next()
{
  //cerr << "FTI next on " << *curr << " " << currDepth << endl;
  int ans = 1;
  for( ; ; )
    {
      FTreeMap::iterator& fti = maps[currDepth];
      if(fti != curr->subtree.end())
	{
	  curr = (*fti).second;
	  fti++;
	  currDepth++;
	  maps[currDepth] = curr->subtree.begin();
	  break;
	}
      if(curr->auxNd)
	{
	  curr = curr->auxNd;
	  maps[currDepth] = curr->subtree.begin();
	  continue;
	}
      while(curr->ind == AUXIND)
	{
	  curr = curr->back;
	}
      curr = curr->back;
      if(!curr)
	{
	  ans = 0;
	  break;
	}
      currDepth--;
    }
  if(!ans)
    {
      alive_ = 0;
      return ;
    }
  //cerr << "found " << *curr << " " << currDepth << endl;
  if(curr->feats.empty()) next();
}
