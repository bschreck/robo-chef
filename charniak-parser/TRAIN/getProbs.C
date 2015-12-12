/* output of program
(S1 (S (NP (DT the) (NN employee))                     //from cin
     (VP (VB worked) (NP (CD twenty) (NNS hours)))
     (. .)))
(NP (DT the) (NN employee))                           // np we are working on
GP-1 0.981022                                            p(STOP } np)
GP0 0.325311                                             p(DT | ...)
(DT the)
GPH 0.401471                                             p(the | ...)
(NP (CD twenty) (NNS hours))                             other np
GP-1 0.692088                                            p(STOP{...)
     //now for the nextsentence with the determiners reversed
(S1 (S (NP (NN employee))
     (VP (VB worked) (NP (DT the) (CD twenty) (NNS hours)))
     (. .)))
(NP (DT the) (CD twenty) (NNS hours))
GP-1 0.972161
GP0 0.056982
(DT the)
GPH 0.676892
(NP (NN employee))
GP-1 0.0826738
*/

/*
The critical program is:
float getProb(InputTree* tree, int pos, int whichInt);
tree points to the subtree we are working on,
pos is the position of the subconstituent we are adding on to tree
and whichInt indicates which probabililty distribution we are using:
In Feature.h we have defined
LCALC to be the number for the distribution to use when expanding to the left,
and HCALC is for the head.  Note that pos above is not used for HCALC
since tree points to, e.g., (DT the) and there is only one position.
*/


#include "getProbs.h"


/* for a given history, as specified by a tree, for each feature f_i record
   how often it was used. */


//FeatureTree* tRoot = NULL;
extern InputTree* curTree;

extern float unsmoothedPs[MAXNUMFS];
typedef set<string, less<string> > StringSet;


  
void
getProbs::init(ECString path)
{
  Feat::Usage = PARSE;
  addSubFeatureFns();

  ECString tmpA[NUMCALCS] = {"r","h","u","m","l","lm","ru","rm","tt",
			     "s","ww"};

  for(int which = 0 ; which < NUMCALCS ; which++)
    {

      ECString tmp = tmpA[which];      

      Feature::init(path, tmp); 
      if(tmp != "l" && tmp != "h") continue;
      ECString ftstr(path);
      ftstr += tmp;
      ftstr += ".g";
      ifstream fts(ftstr.c_str()); 
      if(!fts) cerr << "could not find " << ftstr << endl;
      assert(fts);
      FeatureTree* ft = new FeatureTree(fts); //puts it in root;
      assert(ft);
     
      Feature::readLam(which, tmp, path);
    }
} 



float
getProb(InputTree* tree, int pos, int whichInt)
{
  Feature::whichInt = whichInt;
  TreeHist treeh(tree, pos);
  int hpos = 0;
  if(tree->subTrees().size() != 0) hpos = headPosFromTree(tree);
  treeh.hpos = hpos;

  FeatureTree* ginfo[MAXNUMFS];
  ginfo[0] = FeatureTree::roots(whichInt);

  int cfi = Feature::conditionedFeatureInt[whichInt];
  int cVal = (*SubFeature::Funs[cfi])(&treeh);

  assert(cVal >= 0);
  curTree = tree; //???;
  //curTree = tree; //???;
  int numFs = Feature::total[whichInt];

  assert(whichInt == Feature::whichInt);
  for(int i = 1 ; i <= numFs ; i++)
    processG(1, whichInt, i, ginfo, &treeh, cVal);
  return unsmoothedPs[numFs];
}

//you have probably already initialized these things, but just in case
void initGetProb(string path){
   struct rlimit 	core_limits;
   core_limits.rlim_cur = 0;
   core_limits.rlim_max = 0;
   setrlimit( RLIMIT_CORE, &core_limits );

   Term::init(path);
   readHeadInfo(path);
   Pst pst(path);
   ClassRule::readCRules(path);

   getProbs::init(path);

}

int
main(int argc, char *argv[])
{

   ECArgs args( argc, argv );
   assert(args.nargs() == 1);
   string  path( args.arg( 0 ) );
   initGetProb(path);

   int i;
   for(i = 0 ; i < 2 ; i++)
     {
       InputTree testData;
       cin >> testData;
       cout << "\n\n" << testData << endl;
       InputTree* e0 = &testData;
       e0 = e0->subTrees().front();
       assert(e0);
       InputTree *e1, *e2, *e3;
       e1 = e0->subTrees().front();
       InputTree *e4, *e5;
       InputTreesIter iti = e0->subTrees().begin();
       iti++;
       e4 = (*iti);
       e5 = e4->subTrees().back();
       InputTree* temp;
       if(i == 1)
	 {
	   temp = e1;
	   e1 = e5;
	   e5 = temp;
	 }
       assert(e1);
       cout << *e1 << endl;
       cout << "GP-1 " << getProb(e1, -1, LCALC) << endl;
       cout << "GP0 " << getProb(e1, 0, LCALC) << endl;
       e2 = e1->subTrees().front();
       assert(e2);
       cout << *e2 << endl;
       cout << "GPH " << getProb(e2, 0, HCALC) << endl;
       cout << *e5 << endl;
       cout << "GP-1 " << getProb(e5, -1, LCALC) << endl;
     }
}



