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
#include "Pst.h" //???;

void setNonTermInts();

/* read through wsj training data.
   compute p(x is head of NT | pos(x) =t) and put it in pTgNt.txt
   also compute ending info for pos's ie. p(ending(x) | pos(x)).
   goes at the end of the same file */

//#define Word_p const Word*

int                 data[MAXNUMTS][MAXNUMNTS];
int numEndings = 0;

typedef map<ECString,int, less<ECString> > endMap;
endMap endData[MAXNUMTS];
int                 numTerm[MAXNUMTS];

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

const Term*
addWwData(InputTree* tree)
{
  bool okSit = true;

  if( tree->word() != ""  )
    {
      ECString wTagNm = tree->term();
      const Term* trm = Term::get(wTagNm);
      int lhsInt = trm->toInt();
      if(trm->openClass()) 
	{
	  ECString hdLexU(tree->word());
	  char temp[512];
	  ECString hdLex(toLower(hdLexU.c_str(),temp));
	  int len = hdLex.length();
	  if(len >= 3)
	    {
	      ECString e(hdLex,len-2,2);
	      //ECString e = hdLex;
	      // if the current count for lhs and e == 0, this is new;
	      const WordInfo* wi = Pst::get(hdLex); //???;
	      if(!wi)
		{
		  assert(wi);
		}
	      if(wi->c() <= 4) 
		{
		  incrEndData(lhsInt, e);
		  numTerm[lhsInt]++;
		}
	    }
	}
      return trm;
    }
  ECString fixedTerm(tree->term());
  if(fixedTerm == "") fixedTerm = "S1";
  const Term* lhs = Term::get(fixedTerm);
  /* If we cannot recognize the term, don't abort, just warn and do not
     create a rule here or one level up. */
  if(!lhs)
    {
      lhs = Term::get("GARBAGE");
      okSit = false;
      cerr << "Garbage term: " << tree->term() << endl;
    }

  InputTrees& st = tree->subTrees();
  InputTrees::iterator  subTreeIter= st.begin();
  InputTree  *subTree;
  for( ; subTreeIter != st.end() ; subTreeIter++ )
    {
      subTree = *subTreeIter;
      addWwData(subTree);
    }
  int lhsInt = lhs->toInt();
  int k, l;
  k = Term::get(tree->headTree()->term())->toInt();
  l = lhsInt - 1 - Term::lastTagInt();
  data[k][l]++;
  return lhs;
}

int
main(int argc, char *argv[])
{
  ECArgs args( argc, argv );
  assert(args.nargs() == 1);
  ECString path(args.arg(0));
  cerr << "At start of pTgNt" << endl;

  for(int n = 0 ; n < MAXNUMTS ; n++)
    numTerm[n] = 0;

  ECString resultsString(path);
  resultsString += "endings.txt";

  Term::init( path );  
  if(args.isset('L')) Term::Language = args.value('L');
  readHeadInfo(path);
  Pst pst(path);

  int sentenceCount = 0;
  int wordCount = 0;
  int processedCount = 0;

  int i, j;
  for(i = 0 ; i < MAXNUMTS ; i++)
    for(j = 0 ; j < MAXNUMNTS ; j++)
      data[i][j] = 0;

  i = 0;
  while(cin)
    {
      if(i%10000 == 0) cerr << i << endl;
      //if(i > 1000) break;
      InputTree  parse;
      cin >> parse;
      if(!cin) break;
      if(parse.length() == 0) break;
      const Term* resTerm = addWwData(&parse);
      processedCount++;
      wordCount += parse.length();
      i++;
    }
  ofstream     resultsStream(resultsString.c_str());
  assert(resultsStream);
  int  totNt[MAXNUMTS];
  for(i = 0 ; i < MAXNUMTS ; i++) totNt[i] = 0;
  for(i = 0 ; i <= Term::lastTagInt() ; i++)
    {
      for(j = 0 ; j < (Term::lastNTInt() - Term::lastTagInt()) ; j++)
	totNt[j] += data[i][j];
    }
  resultsStream << numEndings << "\n";
  for(i = 0 ; i < MAXNUMTS ; i++)
    {
      endMap::iterator emi = endData[i].begin();
      for( ; emi != endData[i].end() ; emi++)
	{
	  ECString ending = (*emi).first;
	  int cnt = (*emi).second;
	  resultsStream << i << "\t" << ending << "\t"
			<< (float) cnt / (float) numTerm[i]
			<< endl;
	    //<< "\n";
	}
    }
  return 0;
}
