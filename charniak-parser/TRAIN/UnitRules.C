/*
 * Copyright 2005 Brown University, Providence, RI.
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

#include "UnitRules.h"
#include "Term.h"

void
UnitRules::
init()
{
  numRules_ = 0;
  for(int i = 0 ; i < MAXNUMNTS ; i++)
    for(int j = 0 ; j < MAXNUMNTS ; j++) treeData_[i][j] = 0;
}

void
UnitRules::
readData(ECString path)
{
  ECString fl(path);
  fl += "unitRules.txt";
  ifstream data(fl.c_str());
  assert(data);
  int lim = Term::lastNTInt() - Term::lastTagInt();
  for(numRules_ = 0 ; numRules_ < lim ; numRules_++)
    {
      if(!data) break;
      data >> unitRules[numRules_];
    }
}

void
UnitRules::
gatherData(InputTree* tree)
{
  const Term* trm = Term::get(tree->term());
  assert(trm);
  int parInt = trm->toInt();
  int rparI = parInt-( Term::lastTagInt() + 1);
  InputTreesIter iti = tree->subTrees().begin();
  int len = tree->subTrees().size();
  for( ; iti != tree->subTrees().end() ; iti++)
    {
      InputTree* stree = (*iti);
      if(len == 1)
	{
	  const Term* strm = Term::get(stree->term());
	  if(strm->terminal_p()) continue;
	  assert(strm);
	  int chiInt = strm->toInt();
	  if(chiInt == parInt) continue;
	  int rchiI = chiInt -( Term::lastTagInt() + 1);
	  treeData_[rparI][rchiI]++;
	  //cerr << "TD " << parInt<<" " << chiInt << " " << treeData_[rparI][rchiI] << endl;
	}
      gatherData(stree);
    }
}

  
bool
UnitRules::
badPair(int par, int chi)
{
  bool seenPar = false;
  for(int i = 0 ; i < numRules_ ; i++)
    {
      int nxt = unitRules[i];
      if(nxt == chi) return !seenPar;
      if(nxt == par) seenPar = true;
    }
  return true;
}

void
recMark(int p, int c, int bef[MAXNUMNTS][MAXNUMNTS], int lim)
{
  assert(bef[p][c] != 0);
  if(bef[p][c] >= 1) return;
  bef[p][c] = 1;
  bef[c][p] = 0;
  for(int k = 0 ; k < lim ; k++)
    {
      if(bef[c][k] > 0) recMark(p, k, bef, lim);
      if(bef[k][p] > 0) recMark(k, c, bef, lim);
    }
}

void
UnitRules::
setData(ECString path)
{
  int p,c,k;
  int bef[MAXNUMNTS][MAXNUMNTS];
  for(p = 0 ; p < MAXNUMNTS ; p++)
    {
      for(c = 0 ; c < MAXNUMNTS ; c++) bef[p][c] = -1;
      bef[p][p] = 0;
    }
  
  int fix = Term::lastTagInt()+1;
  int lim = Term::lastNTInt() - Term::lastTagInt();
  int numToDo = lim*(lim-1);
  int numDone = 0;
  //cerr << lim << " " << numToDo << endl;
  while(numDone < numToDo)
    {
      int bestPar = -1;
      int bestChi = -1;
      int bestVal = -1;
      for(p = 0 ; p < lim ; p++)
	for(c = 0 ; c < lim ; c++)
	  {
	    if(bef[p][c] >= 0) continue;
	    int val =treeData_[p][c];
	    if(val > bestVal)
	      {
		bestVal = val;
		bestPar = p;
		bestChi = c;
	      }
	  }
      if(bestVal <= 3) break;
      //cerr << "NBV " << bestPar+fix << " " << bestChi+fix << " " << bestVal << endl;
      recMark(bestPar, bestChi, bef, lim);
    }
  ECString fl(path);
  fl += "unitRules.txt";
  ofstream data(fl.c_str());
  assert(data);
  for(p = 0 ; p < MAXNUMNTS ; p++)
    for(c = 0 ; c < MAXNUMNTS ; c++)
      if(bef[p][c] > 0) data << p << "\t" << c << "\n";
}
	  
	  

  


