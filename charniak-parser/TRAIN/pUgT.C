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

#include <iostream>
#include <fstream>

#include "ECArgs.h"
#include "ECString.h"
#include "Term.h"
#include "utils.h"
#include "InputTree.h"
#include <map>
#include "headFinder.h"
#include "Pst.h" 

extern bool okFoldSent(int sntNum, int fld, int fOp);
int foldOp = 0;

int posDenoms[MAXNUMTS];
int posUCounts[MAXNUMTS];
int posDashCounts[MAXNUMTS];
int posCounts[MAXNUMTS];
int totCounts[MAXNUMTS];
int posCapCounts[MAXNUMTS];

void setNonTermInts();

/* read through wsj training data.
   compute p(x is head of NT | pos(x) =t) and put it in pTgNt.txt */

InputTree* curSent;

int numEndings = 0;

typedef map<ECString,int, less<ECString> > endMap;
endMap endData[MAXNUMTS];
int                 numTerm[MAXNUMNTS];

void
incrEndData(int lhsInt, ECString e)
{
  endMap::iterator emi = endData[lhsInt].find(e);
  if(emi == endData[lhsInt].end())
    {
      endData[lhsInt][e] = 1;
      numEndings++;
    }
  else
    {
      (*emi).second++;
    }

}

void
addWwData(InputTree* tree)
{
  ECString wTagNm = tree->term();
  const Term* trm = Term::get(wTagNm);
  int lhsInt = trm->toInt();
  totCounts[lhsInt]++;
  if( tree->word() != ""  )
    {
      ECString hdLexU(tree->word());
      char temp[512];
      ECString hdLex(toLower(hdLexU.c_str(),temp));
      int len = hdLex.length();
      const WordInfo* wi = Pst::get(hdLex); //???;
      assert(wi);
      /* Ignore words very close to start of sentence, those
	 that are of length 1, and those who's capitalization is
	 ambiguous. */
      if(tree->start() >= 2 && len > 1
	 &&!(hdLex[0] != hdLexU[0] && hdLex[1] != hdLexU[1]))
	{
	  posCounts[lhsInt]++;
	  if(hdLex[0] != hdLexU[0] && hdLex[1] == hdLexU[1])
	    {
	      posCapCounts[lhsInt]++;
	    }
	}
      posDenoms[lhsInt]++;
      if(wi->c() <= 2)
	{
	  posUCounts[lhsInt]++;
	  char* hyppos =  strpbrk(hdLex.c_str(), "-");
	  if(hyppos) posDashCounts[lhsInt]++;
	}
      return;
    }
  InputTrees& st = tree->subTrees();
  InputTrees::iterator  subTreeIter= st.begin();
  InputTree  *subTree;
  for( ; subTreeIter != st.end() ; subTreeIter++ )
    {
      subTree = *subTreeIter;
      addWwData(subTree);
    }
}

int
main(int argc, char *argv[])
{
  ECArgs args( argc, argv );
  ECString path(args.arg(0));
  cerr << "At start of pUgT" << endl;

  Term::init( path );  
  if(args.isset('L')) Term::Language = args.value('L');
  readHeadInfo(path);
  Pst pst(path);

  int sentenceCount = 0;

  int i, j;
  for(i = 0 ; i < MAXNUMTS ; i++)
    {
      posCounts[i] = 0;
      posCapCounts[i] = 0;
      posDenoms[i] = 0;
      posUCounts[i] = 0;
      posDashCounts[i] = 0;
    }
  for(i = 0 ; i < MAXNUMTS ; i++) totCounts[i] = 0;

  i = 0;
  for( ; ; )
    {
      if(i++%10000 == 1) cerr << i << endl;
      //if(i > 1000) break;
      InputTree  parse;
      cin >> parse;
      //cerr << parse << endl;
      if(parse.length() == 0) break;
      if(!cin) break;
      curSent = &parse;
      addWwData(&parse);
      sentenceCount++;
    }

  ECString resultsString(path);
  resultsString += "pUgT.txt";
  ofstream     resultsStream(resultsString.c_str());
  assert(resultsStream);
  /* we print out p(unknown|tag)    p(Capital|tag)   p(hasDash|tag, unknown)
     note for Capital the denom is different because we ignore the first
     two words of the sentence */
  int nm = Term::lastTagInt()+1;
  for(i = 0 ; i < nm ; i++)
    {
      resultsStream << i << "\t";
      float pugt = 0;
      float pudenom = (float)posDenoms[i];
      if(pudenom > 0) pugt = (float)posUCounts[i]/pudenom;
      resultsStream << pugt << "\t";
      if(posCounts[i] == 0) resultsStream << 0 << "\t";
      else
	resultsStream << (float) posCapCounts[i]/ (float)posCounts[i] << "\t";
      if(posUCounts[i] == 0) resultsStream << 0;
      else resultsStream << (float)posDashCounts[i]/posUCounts[i] ;
      resultsStream << endl;
    }
  ECString resultsString2(path);
  resultsString2 += "nttCounts.txt";
  ofstream     resultsStream2(resultsString2.c_str());
  assert(resultsStream2);
  for(i = 0 ; i <= Term::lastNTInt() ; i++)
    {
      resultsStream2 << i << "\t";
      resultsStream2 << totCounts[i] << "\n";
    }
  return 0;
}
