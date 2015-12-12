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
#include "Term.h"
#include "utils.h"
#include "InputTree.h"
#include "headFinder.h"
#include "headFinderCh.h"
#include "UnitRules.h"
#include "Feature.h"

extern bool okFoldSent(int sntNum, int fld, int fOp);
int foldOp = 0;

void setNonTermInts();

/* read through wsj training data.
   compute p(x is head of NT | pos(x) =t) and put it in pTgNt.txt */


typedef map< int, int, less<int> > PosD;
typedef map<ECString, PosD, less<ECString> > WordMap;
WordMap wordMap;
int                 numTerm[MAXNUMNTS];

void
incrWordData(int lhsInt, ECString wupper)
{
  char temp[128];
  ECString w(toLower(wupper.c_str(), temp));
  numTerm[lhsInt]++;
  WordMap::iterator wmi = wordMap.find(w);
  if(wmi == wordMap.end())
    {
      wordMap[w][lhsInt] = 1;
      return;
    }
  PosD& posd = (*wmi).second;
  PosD::iterator pdi = posd.find(lhsInt);
  if(pdi == posd.end())
    {
      posd[lhsInt] = 1;
    }
  else
    (*pdi).second++;
}


void
addWwData(InputTree* tree)
{
  InputTrees& st = tree->subTrees();
  InputTrees::iterator  subTreeIter= st.begin();
  InputTree  *subTree;
  for( ; subTreeIter != st.end() ; subTreeIter++ )
    {
      subTree = *subTreeIter;
      addWwData(subTree);
    }
  //cerr << *tree << endl;
  if( tree->word() != ""  )
    {
      ECString w = tree->word();
      const Term* trm = Term::get(tree->term());
      assert(trm);
      int trmInt = trm->toInt();
      incrWordData(trmInt, w);
    }
}

int
main(int argc, char *argv[])
{
  ECArgs args( argc, argv );
  assert(args.nargs() == 1);
  ECString path(args.arg(0));
  cerr << "At start of pHsgt" << endl;

  for(int n = 0 ; n < MAXNUMNTS ; n++)
    numTerm[n] = 0;

  Term::init( path );
  if(args.isset('L')) Term::Language = args.value('L');
  readHeadInfo(path);


  int sentenceCount = 0;

  ECString s1lex("^^");
  ECString s1nm("S1");
  int s1Int = Term::get(s1nm)->toInt();
	
  UnitRules ur;
  ur.init();
  while(cin)
    {
      //if(sentenceCount > 4000) break;
      if(sentenceCount%10000 == 0) cerr << sentenceCount << endl;
      InputTree  parse;
      cin >> parse;
      //cerr << parse << endl;
      if(!cin) break;
      int len = parse.length();
      if(len == 0) break;
      else if(len == -1) continue;
      EcSPairs wtList;
      parse.make(wtList); 
      InputTree* par;
      par = &parse;

      addWwData(par);
      incrWordData(s1Int, s1lex);
      ur.gatherData(par);
      sentenceCount++;
    }
  ECString resultsString(path);
  resultsString += "pSgT.txt";
  ofstream     resultsStream(resultsString.c_str());
  assert(resultsStream);

  int numWords = 0;
  resultsStream << "       \n";  //leave space for number of words;
  resultsStream.precision(3);
  ECString lastWord;
  int wordFreq = 0;
  WordMap::iterator wmi = wordMap.begin();
  resultsStream << wordMap.size() << "\n\n";
  for( ; wmi != wordMap.end() ; wmi++)
    {
      ECString w = (*wmi).first;
      resultsStream << w << "\t";
      PosD& posd = (*wmi).second;
      PosD::iterator pdi = posd.begin();
      int count = 0;
      for( ; pdi != posd.end(); pdi++)
	{
	  int posInt = (*pdi).first;
	  int c = (*pdi).second;
	  count += c;
	  float p = (float)c/(float)numTerm[posInt];
	  resultsStream << posInt << " " << p << " ";
	}
      resultsStream << "| " << count << "\n";
    }
  ur.setData(path);
  cerr << "Number of sentences = " << sentenceCount << endl;
  return 1;
}
