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
#include <sys/resource.h>
#include <iostream>
#include <unistd.h>
#include <set>
#include "ECArgs.h"
#include "Feature.h"
#include "FeatureTree.h"
#include "InputTree.h"
#include "headFinder.h"
#include "treeHistSf.h"
#include "TreeHist.h"
#include "Term.h"
#include "Pst.h"
#include "ECString.h"
#include "ClassRule.h"

/* for a given history, as specified by a tree, for each feature f_i record
   how often it was used. */
bool prune = false;
InputTree* curS = NULL;
int c_Val;
int sentenceCount = 0;

extern bool okFoldSent(int sntNum, int fld, int fOp);
int foldOp = 0;

typedef set<ECString, less<ECString> > StringSet;
StringSet wordSet;
bool
inWordSet(int i)
{
  const ECString& w = Pst::fromInt(i);
  if(wordSet.find(w) == wordSet.end()) return false;
  return true;
}

InputTree* sentence[256];
int endPos;
void wordsFromTree(InputTree* tree);

void
makeSent(InputTree* tree)
{
  endPos = 0;
  for(int i = 0 ; i < 256 ; i++) sentence[i] = NULL;
  wordsFromTree(tree);
  if(endPos != tree->finish())
    {
      cerr << endPos << " " << tree->finish() << endl;
      assert(endPos == tree->finish());
    }
}

void
wordsFromTree(InputTree* tree)
{
  if(tree->word() != "")
    {
      sentence[endPos++] = tree;
      return;
    }
  InputTreesIter subTreeIter = tree->subTrees().begin();
  for( ; subTreeIter != tree->subTrees().end() ; subTreeIter++ )
    {
      InputTree* subTree = *subTreeIter;
      wordsFromTree(subTree);
    }
}

int nfeatVs[20];

void
processG(int i, FeatureTree* ginfo[], TreeHist* treeh, int cVal)
{
  Feature* feat = Feature::fromInt(i, Feature::whichInt); 

  /* e.g., g(rtlu) starts from where g(rtl) left off (after tl)*/
  int searchStartInd = feat->startPos;
  FeatureTree* strt = ginfo[searchStartInd];
  assert(strt);
  SubFeature* sf = SubFeature::fromInt(feat->subFeat, Feature::whichInt);
  //  cerr << "pg " << sf->name << endl;
  int nfeatV = (*(sf->fun))(treeh);
  nfeatVs[i] = nfeatV;
  if(nfeatV < 0 && Feat::Usage != PARSE)
    {
      ginfo[i] = NULL;
      return;
    }
  int ufi = sf->usf;
  FeatureTree* histPt = strt->next(nfeatV, feat->auxCnt); 
  assert(histPt);
  ginfo[i] = histPt;
  //cerr << "pg " << i << " " << searchStartInd << " " << nfeatV
  //   << " " << cVal << " " << treeh->pos <<  endl;
  histPt->count++;
  histPt->feats[cVal].cnt()++;
  /*
  if(i == 5 && nfeatVs[1] == 55 && nfeatVs[2] == 13
     && nfeatVs[3] == 55 && nfeatVs[4] == 13 && cVal == 55)
    {
      cerr <<"\n" << nfeatVs[5] << "\n";
      cerr << *treeh->tree->parent() << endl;
    }
  */
}

void
callProcG(TreeHist* treeh)
{
  int i;
  for(i = 0 ; i < 20 ; i++) nfeatVs[i] = -1;
  FeatureTree* ginfo[MAXNUMFS];
  ginfo[0] = FeatureTree::root(); 
  int cVal = (*Feature::conditionedEvent)(treeh);
  if(cVal < 0) return;
  c_Val = cVal;
  for(i = 1 ; i <= Feature::total[Feature::whichInt] ; i++)
    {
      ginfo[i] = NULL;
      processG(i, ginfo, treeh, cVal);
    }
  /*
  if(cVal == 62 && nfeatVs[1] == 69 && nfeatVs[2] == 0 && nfeatVs[9] == 18963)
    {
      cerr << *treeh->tree << "\n";
      cerr << *curS << "\n\n";
    }
    */
      
}

void
gatherFfCounts(InputTree* tree, int inHpos)
{
  InputTrees& st = tree->subTrees();;
  InputTrees::iterator  subTreeIter= st.begin();
  InputTree  *subTree;
  int hpos = 0;
  if(st.size() != 0) hpos = headPosFromTree(tree);
  //cerr << hpos << *tree << endl;
  int pos = 0;
  for( ; subTreeIter != st.end() ; subTreeIter++ )
    {
      subTree = *subTreeIter;
      gatherFfCounts(subTree, pos==hpos ? 1 : 0);
      pos++;
    }
  //cerr << "g " << *tree << endl;
  TreeHist treeh(tree, 0);
  treeh.pos = pos;
  treeh.hpos = hpos;
  const Term* lhsTerm = Term::get(tree->term());
  if(Feature::whichInt == HCALC || Feature::whichInt == UCALC)
    {
      if(!inHpos) callProcG(&treeh);
      return;
    }
  if(lhsTerm->terminal_p())
    {
      if(Feature::whichInt == TTCALC) callProcG(&treeh);
      return;
    }
  if(st.size() == 1 && st.front()->term() == tree->term()) return;
  //cerr << "gff " << *tree << endl;
  //if(tree->term() == "PP" && !st.empty() && st.front()->term() == "VBG")
    //  cerr << *tree << "\n" << *curS << "\n----\n"; //???;
  subTreeIter = st.begin();
  int cVal;
  treeh.pos = -1;
  if(Feature::whichInt == LMCALC) callProcG(&treeh);
  if(Feature::whichInt == LCALC) callProcG(&treeh);
  pos = 0;
  for( ; subTreeIter != st.end() ; subTreeIter++)
    {
      treeh.pos = pos;
      if(pos == hpos && Feature::whichInt == MCALC) callProcG(&treeh);
      if(pos < hpos && Feature::whichInt == LCALC) callProcG(&treeh);
      if(pos > hpos && Feature::whichInt == RCALC) callProcG(&treeh);
      if(pos == hpos && Feature::whichInt == RUCALC) callProcG(&treeh);
      if(pos >= hpos && Feature::whichInt == RMCALC) callProcG(&treeh);
      if(pos <= hpos && Feature::whichInt == LMCALC) callProcG(&treeh);
      pos++;
    }
  //cerr << "gg " << *tree << endl;
  treeh.pos = pos;
  if(Feature::whichInt == RCALC) callProcG(&treeh);
  if(Feature::whichInt == RMCALC) callProcG(&treeh);
}

int
main(int argc, char *argv[])
{
   struct rlimit 	core_limits;
   core_limits.rlim_cur = 0;
   core_limits.rlim_max = 0;
   setrlimit( RLIMIT_CORE, &core_limits );

   ECArgs args( argc, argv );
   assert(args.nargs() == 2);

   ECString  conditionedType( args.arg(0) );
   cerr << "start rCounts " <<  conditionedType << endl;
   if(args.isset('U'))
     {
       Feat::Usage = PARSE;
       cerr << "Special Version for MJ";
     }
   ECString  path( args.arg( 1 ) );

   int minCount = 1;
   if(args.isset('m')) minCount = atoi(args.value('m').c_str());
   FeatureTree::minCount = minCount;
   if(args.isset('M')) Feature::setLM();
   Term::init(path);
   if(args.isset('L')) Term::Language = args.value('L');
   readHeadInfo(path);
   Pst pst(path);
   addSubFeatureFns();

   if(Feature::isLM) ClassRule::readCRules(path);
   Feature::assignCalc(conditionedType);
       
   FeatureTree::root() = new FeatureTree();
   Feature::init(path, conditionedType);
   int ceFunInt = Feature::conditionedFeatureInt[Feature::whichInt];

   Feature::conditionedEvent
     = SubFeature::Funs[ceFunInt];

   sentenceCount = 0;
   //for( ; trainingStream ; sentenceCount++)
   for( ;  ; sentenceCount++)
     {
       if(sentenceCount%10000 == 0)
	 {
	   cerr << "rCounts "
	     << sentenceCount << endl;
	 }
       InputTree     correct;  
       cin >> correct;
       //cerr << correct.length() << endl;
       //cerr << correct << endl;
       if(correct.length() == 0)
	 {
	   break;
	 }
       EcSPairs wtList;
       correct.make(wtList); 
       InputTree* par;
       par = &correct;

       makeSent(par);
       curS = par;
       gatherFfCounts(par, 0);
       if(Feature::whichInt == TTCALC)
	 {
	   list<InputTree*> dummy2;
	   InputTree stopInputTree(par->finish(),par->finish(),"","STOP","",
				   dummy2,NULL,NULL);
	   TreeHist treeh(&stopInputTree,0);
	   treeh.hpos = 0;
	   callProcG(&treeh);
	 }
     }
   ECString resS(path);
   resS += conditionedType;
   resS += ".ff";
   ofstream res(resS.c_str());
   FTreeMap& fts = FeatureTree::root()->subtree;
   FTreeMap::iterator fti = fts.begin();
   //cerr << "Printing to " << resS << endl;
   if(!res)
     {
       cerr << "Could not print to"  << resS;
       assert(res);
     }
  for( ; fti != fts.end() ; fti++)
     {
       int asVal = (*fti).first;
       (*fti).second->printFTree(asVal, res);
     }
   cout << "Total params for " << conditionedType << " = "
	<< FeatureTree::totParams << endl;
   cout << "Number of Sentences = " << sentenceCount << endl;
   cout << "Done: " << (int)sbrk(0) << endl;
}
