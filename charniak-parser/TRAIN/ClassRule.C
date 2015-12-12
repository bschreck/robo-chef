
#include "ClassRule.h"
#include "Term.h"
#include "headFinder.h"
#include "Feature.h"

extern InputTree* tree_2rel_tree(TreeHist* treeh);
extern InputTree* tree_sibling_tree(TreeHist* treeh);
extern InputTree* tree_grandparent_tree(TreeHist* treeh);
extern InputTree* tree_parent_tree(TreeHist* treeh);
extern int headFromTree(InputTree* tree);
extern int tree_term(TreeHist* treeh);
extern int tree_parent_term(TreeHist* treeh);
extern int tree_pos(TreeHist*treeh);

vector<ClassRule>  ClassRule::rBundles3_[100][50];
vector<ClassRule>  ClassRule::rBundles2_[100][50];
vector<ClassRule>  ClassRule::rBundlesm_[100][50];


CRuleBundle&
ClassRule::
getCRules(TreeHist* treeh, int wh)
{
  int modu = Term::stopTerm->toInt();
  int d = tree_term(treeh);
  int m;
  if(wh == MCALCRULES)
    {
      m=tree_pos(treeh);
    }
    
  else m = tree_parent_term(treeh);
  if(wh==3) return rBundles3_[d][m-modu];
  else if(wh == MCALCRULES) return rBundlesm_[d][m];
  else return rBundles2_[d][m-modu];
}

int
posFromTree(InputTree* me, InputTree* par)
{
  InputTreesIter iti = par->subTrees().begin();
  int ans = 0;
  for( ; iti != par->subTrees().end() ; iti++)
    {
      if(*iti == me) return ans;
      ans++;
    }
  error("Never here" );
}

InputTree*
ClassRule::
apply(TreeHist* treeh)
{
  InputTree* tree = treeh->tree;
  InputTree* trdTree;
  int wI = Feature::whichInt;
  //cerr << "AP " << *this << endl;
  if(rel_ == 0)
    {
      if(wI == HCALC || wI == UCALC)
	trdTree = tree_grandparent_tree(treeh);
      else
	{
	  if(Feature::whichInt == UCALC) trdTree = tree_parent_tree(treeh);
	  else trdTree = tree->parent();
	}
    }
  else if(rel_ == 3)
    {
      trdTree = tree_parent_tree(treeh);
    }
  else
    {
      InputTree* par = tree->parent();
      assert(par);
      int hpos = headPosFromTree(par);
      int pos = posFromTree(tree, par);
      TreeHist nth(par, pos);
      nth.hpos = hpos;
      if(rel_ == 1) trdTree = tree_sibling_tree(&nth);
      else if(rel_ == 2) trdTree = tree_2rel_tree(&nth);
      else error("bad relation");
    }
  if(!trdTree)
    {
      return NULL;
    }
  const Term* trm = Term::get(trdTree->term());
  assert(trm);
  if(t_ != trm->toInt()) return NULL;
  return trdTree;
}

  
void
ClassRule::
readCRules(ECString path)
{
  ECString flnm = path;
  flnm += "rules.txt";
  ifstream is(flnm.c_str());
  int wh = 2;

  assert(is);

  int modu = Term::stopTerm->toInt();
  ECString tmp;
  for( ; ; )
    {
      int d, m, r, t;
      is >> tmp;
      if(tmp == "Thirds:")
	{
	  wh = 3;
	  continue;
	}
      //cerr << "T1 " << tmp << endl;
      if(!is) break;
      d = Term::get(tmp)->toInt();
      is >> tmp;
      m = Term::get(tmp)->toInt();
      is >> r;
      r--;
      is >> tmp;
      t = Term::get(tmp)->toInt();
      assert(is);
      ClassRule cr(d,m,r,t);
      //cerr << "RR " << cr << endl;
      if(wh == 3) rBundles3_[d][m-modu].push_back(cr);
      else rBundles2_[d][m-modu].push_back(cr);
    }
  flnm = path;
  flnm += "rules.m";
  ifstream ism(flnm.c_str());
  if(!ism) return;
  ism >> tmp; // all thirds;
  for( ; ; )
    {
      ECString tmp;
      int d, m, r, t;
      ism >> tmp;
      //cerr << "T1 " << tmp << endl;
      if(!ism) break;
      d = Term::get(tmp)->toInt();
      ism >> tmp;
      m = Term::get(tmp)->toInt();
      ism >> tmp;
      t = Term::get(tmp)->toInt();
      assert(ism);
      ClassRule cr(d,m,0,t);
      rBundlesm_[d][m].push_back(cr);
    }
}
      
