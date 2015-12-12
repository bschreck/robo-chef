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

#include <fstream>
#include <iostream>
#include "Feature.h"
#include "ECString.h"

SubFeature* SubFeature::array_[MAXNUMCALCS][MAXNUMFS];
int SubFeature::total[MAXNUMCALCS];
int (*SubFeature::Funs[MAXNUMFS])(TreeHist*);
int (*SubFeature::PRFuns[MAXNUMCALCS])(TreeHist*);
int      SubFeature::ufArray[MAXNUMCALCS][MAXNUMFS];
int      SubFeature::splitPts[MAXNUMCALCS][MAXNUMFS];
Feature* Feature::array_[MAXNUMCALCS][MAXNUMFS];
float*     Feature::lambdas_[MAXNUMCALCS][MAXNUMFS];
int Feature::total[MAXNUMCALCS];
int  Feature::whichInt;
int Feature::assumedFeatVal;
int (*Feature::conditionedEvent)(TreeHist*);
int (*Feature::assumedSubFeat)(TreeHist*);
FTypeTree Feature::ftTree[MAXNUMCALCS];
FTypeTree* Feature::ftTreeFromInt[MAXNUMCALCS][MAXNUMFS];
int Feature::conditionedFeatureInt[MAXNUMCALCS];
int Feature::numCalcs = 9;
bool Feature::isLM = false;
int MinCount = 5;

//JT from prev version that went with getProbs
float Feature::logFacs[NUMCALCS][MAXNUMFS];



void
Feature::
assignCalc(ECString conditioned)
{

  if(conditioned == "h") whichInt = HCALC;
  else if(conditioned == "u") whichInt = UCALC;
  else if(conditioned == "r") whichInt = RCALC;
  else if(conditioned == "ru") whichInt = RUCALC;
  else if(conditioned == "rm") whichInt = RMCALC;
  else if(conditioned == "tt") whichInt = TTCALC;
  else if(conditioned == "l") whichInt = LCALC;
  else if(conditioned == "lm") whichInt = LMCALC;
  else if(conditioned == "ww") whichInt = WWCALC;

  //JT added SCALC from prev version
  else if(conditioned == "s") whichInt = SCALC;


  else
    {
      assert(conditioned == "m");
      whichInt = MCALC;
    }
}

void
Feature::
init(ECString& path, ECString& conditioned)
{
  assignCalc(conditioned);
  int f;
  for(f = 0 ; f < MAXNUMFS ; f++)
    {
      float* vec = new float[15];
      lambdas_[whichInt][f] = vec;
      for(int k = 0 ; k < 15 ; k++) vec[k] = 0.0;
    }

  ECString dataString(path);
  dataString += "featInfo.";
  dataString += conditioned;
  ifstream dataStrm(dataString.c_str());
  if(!dataStrm)
    {
      cerr << "Could not find " << dataString << endl;
      assert(dataStrm);
    }

  int conditionedInt;
  dataStrm >> conditionedInt;
  conditionedFeatureInt[whichInt] = conditionedInt;

  int auxCnts[MAXNUMFS];
  int posNums[MAXNUMFS];
  int i, j;
  for(i = 0 ; i < MAXNUMFS ; i++) auxCnts[i] = 0;
  Feature::ftTreeFromInt[whichInt][0] = &(Feature::ftTree[whichInt]);
  int num;
  for(num = 0 ;  ; num++)
    {
      int n, subf, pos, cpr;
      ECString nm;
      ECString tmp;
      dataStrm >> tmp;
      if(tmp == "--") break;
      n = atoi(tmp.c_str());
      dataStrm >> nm;
      dataStrm >> subf;
      dataStrm >> pos;
      dataStrm >> tmp;
      
      if(tmp == "|")
	cpr = -1;
      else
	{
	  cpr = atoi(tmp.c_str());
	  dataStrm >> tmp;
	  assert(tmp == "|");
	}
      array_[whichInt][n-1] = new Feature(n, nm, subf, pos, cpr);
      array_[whichInt][n-1]->auxCnt = auxCnts[pos];
      auxCnts[pos]++;

      createFTypeTree(Feature::ftTreeFromInt[whichInt][pos], n, whichInt);
    }
  Feature::total[whichInt] = num;
  for(num = 0 ;  ; num++)
    {
      int n, fn;
      ECString nm;
      ECString tmp;
      dataStrm >> tmp;
      if(tmp == "--") break;
      n = atoi(tmp.c_str());
      dataStrm >> nm;
      dataStrm >> fn;
      list<int> featList;
      for( ; ; )
	{
	  dataStrm >> tmp;
	  if(tmp == "|") break;
	  int f = atoi(tmp.c_str());
	  featList.push_back(f);
	}
      SubFeature::fromInt(n, whichInt)
	= new SubFeature(n, nm, fn, featList);
      assert(SubFeature::fromInt(n, whichInt));
    }
  SubFeature::total[whichInt] = num;
  /* set the universal function num on feats from their subfeats */
  for(num = 0 ; num < Feature::total[whichInt] ; num++)
    {
      Feature* f = array_[whichInt][num];
      f->usubFeat = SubFeature::fromInt(f->subFeat,whichInt)->usf;
    }
  /* set up the table from universal subfeat nums to subfeat nums */
  for(num = 0 ; num < MAXNUMFS ; num++)
    SubFeature::ufArray[whichInt][num] = -1;
  for(num = 0 ; num < SubFeature::total[whichInt] ; num++)
    {
      SubFeature* sf = SubFeature::fromInt(num,whichInt);
      SubFeature::ufArray[whichInt][sf->usf] = num;
    }
}

void
Feature::
createFTypeTree(FTypeTree* posftTree, int n, int which)
{
  if(!posftTree->left)
    {
      posftTree->left = new FTypeTree(n);
      Feature::ftTreeFromInt[which][n] = posftTree->left;
      //cerr<< "hanging from " << posftTree->n << " we now have " << n << endl;
    }
  else if(!posftTree->right)
    {
      posftTree->right = new FTypeTree(AUXIND);
      createFTypeTree(posftTree->right, n, which);
    }
  else createFTypeTree(posftTree->right, n, which);
}     

void
Feature::
readLam(int which, ECString tmp, ECString path)
{
  ECString ftstr(path);
  ftstr += tmp;
  ftstr += ".lambdas";
  ifstream fts(ftstr.c_str());
  assert(fts);
  int b,f;
  int tot = Feature::total[which];
  
    /* The standard training programs never read in lambdas.  Only
     getProbs does, and it uses the new bucketing, which uses the
     next for loop
  */

  for(f = 2 ; f <= tot ; f++)
    {
      float logBase;
      
      fts >> logBase;

      /*JT: this used to be  logFacs[f][which] = 1.0/log(logBase), but that
	occurs in bucket
       */
      logFacs[which][f] = logBase;  
    }
  
   
  for(b = 1; b < 15 ; b++)
     {
       int bb;
       if(!fts)
	 {
	   cerr << "Trouble reading lambs for " << which << " in " << ftstr
		<< endl;
	   assert(fts);
	 }
       fts >> bb ;
       //cerr << bb << endl;
       if(bb != b)
	 {
	   cerr << tmp << " " << b << " " << bb << endl;
	   assert(bb == b);
	 }
       for(f = 2; f <= tot ; f++)
	 {
	   float lam;
	   assert(fts);
	   fts >> lam;
	   //cerr << which << " " << f << " " << b << " " << lam << endl;
	   Feature::setLambda(which,f,b,lam);
	 }
     }
} 
