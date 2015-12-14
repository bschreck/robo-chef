
#include <fstream>
#include <sys/resource.h>
#include <iostream>
#include <unistd.h>
#include <set>
#include "ECArgs.h"
#include "Feature.h"
#include "FeatureTree.h"
#include "FeatTreeIter.h"
#include "InputTree.h"
#include "headFinder.h"
#include "headFinderCh.h"
#include "treeHistSf.h"
#include "TreeHist.h"
#include "Term.h"
#include "Pst.h"
#include "ECString.h"

InputTree* curS = NULL;
int numGram = 3;
float ds[4][3];
void finalProbComputation();

typedef set<string, less<string> > StringSet;
StringSet wordSet;
bool
inWordSet(int i)
{
  const string& w = Pst::fromInt(i);
  if(wordSet.find(w) == wordSet.end()) return false;
  return true;
}

InputTree* sentence[256];
int endPos;
void wordsFromTree(InputTree* tree);

int totWords = 0;

void
makeSent(InputTree* tree)
{
  endPos = 0;
  for(int i = 0 ; i < 128 ; i++) sentence[i] = NULL;
  wordsFromTree(tree);
  assert(endPos == tree->finish());
}

void
wordsFromTree(InputTree* tree)
{
  if(tree->word() != "")
    {
      sentence[endPos++] = tree;
      totWords++;
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
incrHistPt(FeatureTree* histPt, int cVal, int i)
{
  //cerr << "ihp " << histPt->ind << " " << histPt->count << endl;
  histPt->count++;
  Feat& feat = histPt->feats[cVal];
  feat.ind()++;
  if(feat.ind() > feat.cnt())
    {
      cerr << "Bad counts " << feat.ind() << " " << feat.cnt() << endl;
      assert(feat.ind() <= feat.cnt());
    }
  if(feat.cnt() > 1) return;
  FeatureTree* par = histPt->back;
  if(par == FeatureTree::root()) return;
  else incrHistPt(par, cVal, i-1);
}

void
processG(int i, FeatureTree* ginfo[], TreeHist* treeh, int cVal)
{
  Feature* feat = Feature::fromInt(i, Feature::whichInt); 

  /* e.g., g(rtlu) starts from where g(rtl) left off (after tl)*/
  int searchStartInd = feat->startPos;
  FeatureTree* strt = ginfo[searchStartInd];
  assert(strt);
  SubFeature* sf = SubFeature::fromInt(feat->subFeat, Feature::whichInt);
  int nfeatV = (*(sf->fun))(treeh);
  nfeatVs[i] = nfeatV; 
  int ufi = sf->usf;
  FeatureTree* histPt = strt->next(nfeatV, feat->auxCnt); 
  assert(histPt);
  ginfo[i] = histPt;
  //cerr << "i" << i << " " << nfeatV << " "
  //   << histPt->ind << " " << histPt->count << " " << cVal << endl;
  histPt->feats[cVal].cnt()++;
  //cerr << "HP " <<   histPt->feats[cVal].cnt() << endl;
  if(i == numGram)
    {
      incrHistPt(histPt, cVal,i);
    }
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
  for(i = 1 ; i <= Feature::total[Feature::whichInt] ; i++)
    processG(i, ginfo, treeh, cVal);
}

void
gatherFfCounts(InputTree* tree, int inHpos)
{
  int wI =Feature::whichInt;
  InputTrees& st = tree->subTrees();;
  InputTrees::iterator  subTreeIter= st.begin();
  InputTree  *subTree;
  int hpos = 0;
  if(st.size() != 0) hpos = headPosFromTree(tree);
  int pos = 0;
  for( ; subTreeIter != st.end() ; subTreeIter++ )
    {
      subTree = *subTreeIter;
      gatherFfCounts(subTree, pos==hpos ? 1 : 0);
      pos++;
    }
  //cerr << "g " << inHpos << " " << *tree << endl;
  //cerr << "t " << wI << endl;
  TreeHist treeh(tree, 0);
  treeh.pos = pos;
  treeh.hpos = hpos;
  const Term* lhsTerm = Term::get(tree->term());
  if(wI == HCALC || wI == UCALC)
    {
      if(!inHpos)
	{
	  callProcG(&treeh);
	}
      return;
    }
  if(lhsTerm->terminal_p())
    {
      if(wI == TTCALC || wI == WWCALC) callProcG(&treeh);
      return;
    }
  if(st.size() == 1 && st.front()->term() == tree->term()) return;
  //cerr << "gff " << *tree << endl;
  //if(tree->term() == "PP" && !st.empty() && st.front()->term() == "VBG")
    //  cerr << *tree << "\n" << *curS << "\n----\n"; //???;
  subTreeIter = st.begin();
  int cVal;
  treeh.pos = -1;
  if(wI == LMCALC) callProcG(&treeh);
  if(wI == LCALC) callProcG(&treeh);
  pos = 0;
  for( ; subTreeIter != st.end() ; subTreeIter++)
    {
      treeh.pos = pos;
      if(pos == hpos && wI == MCALC) callProcG(&treeh);
      if(pos < hpos && wI == LCALC) callProcG(&treeh);
      if(pos > hpos && wI == RCALC) callProcG(&treeh);
      if(pos == hpos && wI == RUCALC) callProcG(&treeh);
      if(pos >= hpos && wI == RMCALC) callProcG(&treeh);
      if(pos <= hpos && wI == LMCALC) callProcG(&treeh);
      pos++;
    }
  treeh.pos = pos;
  if(wI == RCALC) callProcG(&treeh);
  if(wI == RMCALC) callProcG(&treeh);
  //cerr << "gg " << *tree << endl;
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
   if(args.isset('N')) numGram = atoi(args.value('N').c_str());
   Feature::setLM();
   if(args.isset('L')) Term::Language = args.value('L');
   string  path( args.arg( 1 ) );
   if(Term::Language == "Ch") readHeadInfoCh(path);
   else readHeadInfo(path);

   string  conditionedType( args.arg(0) );
   cerr << "start kn3Counts " <<  conditionedType << endl;
   int minCount = 1;
   if(args.isset('m')) minCount = atoi(args.value('m').c_str());
   Feat::Usage = KNCOUNTS;
   FeatureTree::minCount = minCount;

   Term::init(path);
   readHeadInfo(path);
   Pst pst(path);
   addSubFeatureFns();

   Feature::assignCalc(conditionedType);
       
   FeatureTree::root() = new FeatureTree();
   Feature::init(path, conditionedType);
   int wI = Feature::whichInt;
   int ceFunInt = Feature::conditionedFeatureInt[wI];

   Feature::conditionedEvent
     = SubFeature::Funs[ceFunInt];
   string trainingString( path );

   int sentenceCount = 0;
   for( ; ; sentenceCount++)
     {
       if(sentenceCount%10000 == 1)
	 {
	   cerr << "rCounts "
	     << sentenceCount << endl;
	 }
       InputTree     correct;  
       cin >> correct;
       //if(sentenceCount > 1000) break;
       if(correct.length() == 0) break;
       //cerr <<sentenceCount << correct << endl;
       EcSPairs wtList;
       correct.make(wtList); 
       InputTree* par;
       int strt = 0;
       par = &correct;

       makeSent(par);
       curS = par;
       gatherFfCounts(par, 0);
       if(wI == TTCALC || wI == WWCALC)
	 {
	   list<InputTree*> dummy2;
	   InputTree stopInputTree(par->finish(),par->finish(),
				   wI==TTCALC ? "" : "^^",
				   "STOP","",
				   dummy2,NULL,NULL);
	   stopInputTree.headTree() = &stopInputTree;
	   TreeHist treeh(&stopInputTree,0);
	   treeh.hpos = 0;
	   callProcG(&treeh);
	 }
     }
   finalProbComputation();
   string resS(path);
   resS += conditionedType;
   resS += ".g";
   ofstream res(resS.c_str());
   assert(res);
   FTreeMap& fts = FeatureTree::root()->subtree;
   FTreeMap::iterator fti = fts.begin();
   for( ; fti != fts.end() ; fti++)
     {
       int asVal = (*fti).first;
       (*fti).second->printFTree(asVal, res);
     }
   res.close();
   cout << "Tot words: " << totWords << endl;
   cout << "Total params for " << conditionedType << " = "
	<< FeatureTree::totParams << endl;
   cout << "Done: " << (int)sbrk(0) << endl;
}



int counts13[4][4];

int
level(FeatureTree* tree)
{
  int i;
  for( i = 0 ; ; i++)
    {
      FeatureTree* par = tree->back;
      if(!par) return i;;
      tree = par;
    }
}
  
void
finalProbComputation()
{
  int i,j;
  for(i = 0 ; i < numGram; i++) for (j = 0 ; j < 4 ; j++) counts13[i][j] = 0;
  FeatureTree* root = FeatureTree::roots(Feature::whichInt);
  assert(root);
  FeatTreeIter fi(root);
  FeatureTree* ft;
  /*
    first compute the Ds (discounts) for counts of 1, 2, and 3-up
  */
  for( ; fi.alive() ; fi.next() )
    {
      ft = fi.curr;
      int lev = fi.depth();
      if(lev < 2) continue;
      FeatMap& fm = ft->feats;
      FeatMap::iterator fmi = fm.begin();
      for( ; fmi != fm.end() ;fmi++)
	{
	  Feat& f = (*fmi).second;
	  int cnt = f.cnt();
	  if(cnt > 4) continue;
	  counts13[lev-2][cnt-1]++;
	}
    }
  for(i = 0 ; i <= numGram-2 ; i++)
    {
      float n1 = (float)counts13[i][0];
      float n2 = (float)counts13[i][1];
      float n3 = (float) counts13[i][2];
      float n4 = (float) counts13[i][3];
      float tmp = n1/(n1+2*n2);
      ds[i][0] = 1-(2*tmp*(n2/n1));
      ds[i][1] = 2-(3*tmp*(n3/n2));
      ds[i][2] = 3-(4*tmp*(n4/n3));
      for(j = 0 ; j < 3 ; j++)
	{
	  cerr << "ds " << i << " " << j << " " << ds[i][j] << endl;
	  cerr << n1 << "\t"<< n2 << "\t"<< n3 << "\t"<< n4 <<endl;
	  assert(ds[i][j] >= 0);
	  assert(ds[i][j] < j+1);
	}
    }
  FeatTreeIter fi2(root);
  /*
    Here we go through, and for each probability, actually compute it
    by p = (counts-D)/(total counts)
  */
  for( ; fi2.alive() ; fi2.next())
    {
      ft = fi2.curr;
      int lev = fi2.depth();
      if(lev < 1) continue;
      FeatMap& fm = ft->feats;
      FeatMap::iterator fmi = fm.begin();
      float totDel = 0;
      for( ; fmi != fm.end() ;fmi++)
	{
	  Feat& f = (*fmi).second;
	  float ind = (float)f.ind();
	  float denom = ft->count;
	  int wh = f.ind()-1;
	  if(ind > 2) wh = 2;
	  float dsl = 0;
	  if(lev > 1) dsl = ds[lev-2][wh]; //???;
	  float num = ind-dsl;
	  if(num < 0)
	    {
	      cerr << ind << " " << dsl << endl;
	      assert(num >= 0);
	    }
	  if(denom <= 0)
	    {
	      cerr << "Bad denom: " << denom << ft->ind
		   << " " << ft->featureInt << " " << lev << endl;
	      assert(denom > 0);
	    }
	  totDel += dsl;
	  f.g() = num/denom;
	  if(f.g() > 1)
	    {
	      cerr << "LDN " << lev << " " << denom << " " << num << endl;
	      assert(f.g() <= 1);
	    }
	  //cerr << "fg " << f.g() << endl;
	}
      ft->count = (int)((1000*totDel)/ft->count);
      assert(ft->count <= 1000);
    }
}
