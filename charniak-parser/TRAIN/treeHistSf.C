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

#include "TreeHist.h"
#include "treeHistSf.h"
#include "Term.h"
#include "Feature.h"
#include "InputTree.h"
#include "Pst.h"
#include "headFinder.h"
#include "Feat.h"
#include "ccInd.h"
#include "ClassRule.h"

int stopTermInt;
int nullWordInt;
extern InputTree* sentence[256];
extern int endPos;
extern int c_Val;
InputTree* tree_find(TreeHist* treeh, int n);
InputTree* tree_ruleTree(TreeHist* treeh, int ind);


InputTree*
tree_grandparent_tree(TreeHist* treeh)
{
  InputTree* tree = treeh->tree;
  InputTree* pt = tree->parent();
  if(!pt) return NULL;
  if(pt->headTree() == tree->headTree()) return NULL;
  pt = pt->parent();
  if(!pt) return NULL;
  if(pt->headTree() == tree->headTree()) return NULL;
  return pt;
}


InputTree*
tree_sibling_tree(TreeHist* treeh)
{
  int pos = treeh->pos;
  int hp = treeh->hpos;
  InputTree* tr = NULL;
  //cerr << "tst " << pos << " " << hp << *treeh->tree << endl;
  if(pos > hp+1) tr = tree_find(treeh, -1);
  else if(pos < hp-1) tr = tree_find(treeh, 1);
  return tr;
}

InputTree*
tree_2rel_tree(TreeHist* treeh)
{
  //cerr << "t1r " << *treeh->tree << endl;
  int pos = treeh->pos;
  int hpos = treeh->hpos;
  if(pos == hpos || pos < hpos-1 || pos > hpos+1) return NULL;
  //cerr << "t2r " << *treeh->tree << endl;
  InputTree* sib;
  if(pos < hpos)
    {
      sib = tree_find(treeh, +1);
      int sibhp = headPosFromTree(sib);
      InputTree* sibch;
      if(sibhp > 0)
        {
          sibch = sib->subTrees().front();
        }
      else if(sib->subTrees().size() < 2) return NULL;
      else
        {
          InputTreesIter iti = sib->subTrees().begin();
          iti++;
          sibch = *iti;
        }
      return sibch;
    }
  else
    {
      sib = tree_find(treeh, -1);
      int sibhp = headPosFromTree(sib);
      InputTree* sibch;
      if(sibhp < sib->subTrees().size()-1)
        {
          sibch = sib->subTrees().back();
        }
      else if(sib->subTrees().size() < 2) return NULL;
      else
        {
          InputTrees::reverse_iterator iti = sib->subTrees().rbegin();
          iti++;
          sibch = *iti;
        }
      return sibch;
    }
}



InputTree*
tree_parent_tree(TreeHist* treeh)
{
  InputTree* tree = treeh->tree;
  InputTree* pt = tree->parent();
  if(!pt) return NULL;
  if(pt->headTree() == tree->headTree()) return NULL;
  return pt;
}

int
is_effEnd(InputTree* tree, InputTree* child)
{
  if(!tree) return 1;
  const Term* trm = Term::get(tree->term());
  if(trm->isRoot()) return 1;
  InputTreesIter iti = tree->subTrees().begin();
  for( ; ; iti++)
    {
      assert(iti != tree->subTrees().end());
      InputTree* nxt = (*iti);
      assert(nxt);
      if(nxt != child) continue;
      iti++;
      if(iti == tree->subTrees().end())
        return is_effEnd(tree->parent(),tree);
      nxt = (*iti);
      ECString ntrmNm = nxt->term();
      const Term* ntrm = Term::get(ntrmNm);
      if(ntrm== Term::stopTerm)
        return is_effEnd(tree->parent(),tree);
      if(ntrm->isColon() || ntrm->isFinal()) return 1;
      if(ntrm->isComma()) return 0;
      iti++;
      if(iti == tree->subTrees().end()) return 0;
      nxt = (*iti);
      if(nxt->term() == "''") return 1;
      return 0;
    }
  error("should not get here");
  return 0;
}

int
tree_term(TreeHist* treeh)
{
  InputTree* tree = treeh->tree;
  const ECString& trmStr  = tree->term();
  const Term* trm = Term::get(trmStr);
  assert(trm);
  return trm->toInt();
}

int
tree_parent_term(TreeHist* treeh)
{
  InputTree* tree = treeh->tree;
  static int s1int = 0;
  if(!s1int)
    {
      ECString s1nm("S1");
      s1int = Term::get(s1nm)->toInt();
    }
  InputTree* par = tree->parent();
  if(!par) return s1int;
  const ECString& trmStr  = par->term();
  const Term* trm = Term::get(trmStr);
  assert(trm);
  assert(!trm->terminal_p());
  return trm->toInt();
}

int
toBe(const ECString& parw)
{
  if(parw == "was" || parw == "is" || parw == "be" || parw == "been"
     || parw == "are" || parw == "were" || parw == "being")
    return 1;
  else return 0;
}

int
tree_parent_pos(TreeHist* treeh)
{
  static int stopint = 0;
  InputTree* tree = treeh->tree;
  if(!stopint)
    {
      ECString stopnm("STOP");
      stopint = Term::get(stopnm)->toInt();
    }
  InputTree* par = tree->parent();
  if(!par) return stopint;

  const ECString& trmStr  = par->hTag();
  const Term* trm = Term::get(trmStr);
  assert(trm);
  if(!trm->terminal_p())
    {
      cerr << "Bad head Part of Speech: " << *trm << " in " <<endl;
      cerr << *tree << endl;
      assert(trm->terminal_p());
    }
  int ans = trm->toInt();
  if(ans < 2 && toBe(par->head())) return 48;
  return ans;
}

int
tree_term_before(TreeHist* treeh)
{
  static int stopint = 0;
  if(!stopint)
    {
      ECString stopnm("STOP");
      stopint = Term::get(stopnm)->toInt();
    }
  InputTree* tree = treeh->tree;
  InputTree* par = tree->parent();
  if(!par) return stopint;
  InputTreesIter iti = par->subTrees().begin();
  for( ; iti != par->subTrees().end() ; iti++ )
    {
      InputTree* st = *iti;
      if(st != tree) continue;
      if(iti == par->subTrees().begin()) return stopint;
      iti--;
      st = *iti;
      const ECString& trmStr  = st->term();
      const Term* trm = Term::get(trmStr);
      assert(trm);
      return trm->toInt();
    }
  error("Should never get here");
  return -1;
}

int
tree_term_after(TreeHist* treeh)
{
  static int stopint = 0;
  if(!stopint)
    {
      ECString stopnm("STOP");
      stopint = Term::get(stopnm)->toInt();
    }
  InputTree* tree = treeh->tree;
  InputTree* par = tree->parent();
  if(!par) return stopint;
  InputTreesIter iti = par->subTrees().begin();
  for( ; iti != par->subTrees().end() ; iti++ )
    {
      InputTree* st = *iti;
      if(st != tree) continue;
      iti++;
      if(iti == par->subTrees().end()) return stopint;
      st = *iti;
      const ECString& trmStr  = st->term();
      const Term* trm = Term::get(trmStr);
      assert(trm);
      return trm->toInt();
    }
  error("Should never get here");
  return -1;
}

int
tree_pos(TreeHist* treeh)
{
  InputTree* tree = treeh->tree;
  const ECString& trmStr  = tree->hTag();
  const Term* trm = Term::get(trmStr);
  assert(trm);
  if(!trm->terminal_p())
    {
      cerr << "Bad head Part of Speech: " << *trm << " in " <<endl;
      cerr << *tree << endl;
      assert(trm->terminal_p());
    }
  return trm->toInt();
}

int
headFromTree(InputTree* tree)
{

  char temp[512];
  string wrdStr(toLower(tree->head().c_str(), temp));
  const WordInfo* wi = Pst::get(wrdStr);
  if(!wi)
    {
      if(Feat::Usage == PARSE) return -1;
      cerr << "Could not find " << wrdStr << endl;
      assert(wi);
    }
  int ans = wi->toInt();
  assert(ans >= 0);
  return ans;
}

int
headFromParTree(InputTree* tree)
{
  InputTree* pt = tree->parent();
  if(!pt) return nullWordInt;
  else return headFromTree(pt);
}

int
tree_head(TreeHist* treeh)
{
  return headFromTree(treeh->tree);
}

int
tree_parent_head(TreeHist* treeh)
{
  InputTree* specTree = NULL;
  if(Feature::isLM) specTree = tree_ruleTree(treeh,2);
  if(specTree) return headFromTree(specTree);
  else return headFromParTree(treeh->tree);
}

int
tree_true(TreeHist* treeh)
{
  return 1;
}

int
tree_grandparent_term(TreeHist* treeh)
{
  static int s1int = 0;
  if(!s1int)
    {
      ECString s1nm("S1");
      s1int = Term::get(s1nm)->toInt();
    }
  InputTree* tree = treeh->tree;
  InputTree* par = tree->parent();
  if(!par) return s1int;
  InputTree* gpar = par->parent();
  if(!gpar) return s1int;
  const ECString& trmStr  = gpar->term();
  const Term* trm = Term::get(trmStr);
  assert(trm);
  assert(!trm->terminal_p());
  return trm->toInt();
}

int
tree_grandparent_pos(TreeHist* treeh)
{
  static int stopint = 0;
  if(!stopint)
    {
      ECString stopnm("STOP");
      stopint = Term::get(stopnm)->toInt();
    }
  InputTree* tree = treeh->tree;
  InputTree* par1 = tree->parent();
  if(!par1) return stopint;
  InputTree* par = par1->parent();
  if(!par) return stopint;
  
  const ECString& trmStr  = par->hTag();
  const Term* trm = Term::get(trmStr);
  assert(trm);
  if(!trm->terminal_p())
    {
      cerr << "Bad head Part of Speech: " << *trm << " in " <<endl;
      cerr << *tree << endl;
      assert(trm->terminal_p());
    }
  return trm->toInt();
}

int
tree_grandparent_head(TreeHist* treeh)
{
  InputTree* tree = treeh->tree;
  InputTree* pt = tree->parent();
  static int topInt = -1;
  if(topInt < 0)
    {
      ECString temp("^^");
      topInt = Pst::get(temp)->toInt();
    }
  if(!pt) return topInt;
  pt = pt->parent();
  if(!pt) return topInt;

  char temp[512];
  ECString wrdStr(toLower(pt->head().c_str(),temp));
  const WordInfo* wi = Pst::get(wrdStr);
  if(!wi)
    {
      cerr << *tree << endl;
      assert(wi);
    }
  int ans = wi->toInt();
  assert(ans >= 0);
  return ans;
}

int
tree_ccparent_term(TreeHist* treeh)
{
  static int s1int = 0;
  if(!s1int)
    {
      ECString s1nm("S1");
      s1int = Term::get(s1nm)->toInt();
    }
  assert(treeh);
  InputTree* tree = treeh->tree;
  assert(tree);
  InputTree* par = tree->parent();
  if(!par) return s1int;
  const ECString& trmStr  = par->term();
  const Term* trm = Term::get(trmStr);
  assert(trm);
  int trmInt = trm->toInt();
  if(trmStr != tree->term()) return trmInt; //??? new;
  assert(!trm->terminal_p());
  int ccedtrmInt = ccIndFromTree(par);
  return ccedtrmInt;
}

int
tree_size(TreeHist* treeh)
{
  static int bucs[9] = {1, 3, 6, 10, 15, 21, 28, 36, 999};
  InputTree* tree = treeh->tree;
  int sz = tree->finish() - tree->start();
  for(int i = 0 ; i < 9 ; i++)
    if(sz <= bucs[i]) return i;
  assert("Never get here");
  return -1;
}



int
tree_effEnd(TreeHist* treeh)
{
  InputTree* tree = treeh->tree;
  int pos = tree->finish();
  bool ans;
  if(pos > endPos)
    {
      cout << "Pos > endPos" << endl;
      ans = 0;
    }
  else if(pos == endPos) ans = 1;
  else
    {
      ECString wrd = sentence[pos]->word();
      ECString trm = sentence[pos]->term();
      if(trm == "." || wrd == ";") ans = 1;
      else if((pos+2) > endPos) ans = 0;
      else if(wrd == ",")
	{
	  if(sentence[pos+1]->word() == "''")
	    ans = 1; // ,'' acts like end of sentence;
	  else ans = 0;  //ans = 2 for alt version???
			     }
      else ans = 0;
    }
  return ans;
}

int
tree_ngram(TreeHist* treeh, int n, int l)
{
  static int stopTermInt = -1;
  if(stopTermInt < 0)
    {
      ECString stopStr("STOP");
      const Term* stopTerm = Term::get(stopStr);
      stopTermInt = stopTerm->toInt();
    }

  int pos = treeh->pos;
  int hp = treeh->hpos;
  int m = pos + (n * l);
  if(m < 0) return stopTermInt;
  InputTree* tree = treeh->tree;
  if(m >= tree->subTrees().size()) return stopTermInt;
  if(m > hp && l > 0) return stopTermInt;
  InputTree  *subTree;
  InputTreesIter  subTreeIter = tree->subTrees().begin();
  int i = 0;
  for( ; subTreeIter != tree->subTrees().end() ; subTreeIter++ )
    {
      if(i == m)
	{
	  subTree = *subTreeIter;
	  const Term* trm = Term::get(subTree->term());
	  return trm->toInt();
	}
      i++;
    }
  assert("should never get here");
  return -1;
}

int
tree_vE(TreeHist* treeh)
{
  int v = tree_parent_pos(treeh);
  int e = is_effEnd(treeh->tree->parent(), treeh->tree);
  return v+(e*MAXNUMNTS);
}

int
tree_mE(TreeHist* treeh)
{
  int m = tree_grandparent_term(treeh);
  int e = is_effEnd(treeh->tree->parent(), treeh->tree);
  return m+(e*50);
}


InputTree*
tree_find(TreeHist* treeh, int n)
{
  int pos = treeh->pos;
  int hp = treeh->hpos;
  int m = pos + n;
  assert(m >= 0);
  InputTree* tree = treeh->tree;
  assert(!(m >= tree->subTrees().size()));
  InputTree  *subTree;
  InputTreesIter  subTreeIter = tree->subTrees().begin();
  int i = 0;
  for( ; subTreeIter != tree->subTrees().end() ; subTreeIter++ )
    {
      if(i == m)
	{
	  subTree = *subTreeIter;
	  return subTree;
	}
      i++;
    }
  assert("should never get here");
  return NULL;
}

int
tree_left0(TreeHist* treeh)
{
  return tree_ngram(treeh, 0, 0);
}

int
tree_left1(TreeHist* treeh)
{
  return tree_ngram(treeh, 1, 1);
}

int
tree_left2(TreeHist* treeh)
{
  return tree_ngram(treeh, 2, 1);
}

int
tree_left3(TreeHist* treeh)
{
  return tree_ngram(treeh, 3, 1);
}

int
tree_right1(TreeHist* treeh)
{
  return tree_ngram(treeh, 1, -1);
}

int
tree_right2(TreeHist* treeh)
{
  return tree_ngram(treeh, 2, -1);
}

int
tree_right3(TreeHist* treeh)
{
  return tree_ngram(treeh, 3, -1);
}

int
tree_right4(TreeHist* treeh)
{
  return tree_ngram(treeh, 4, -1);
}

int
tree_left4(TreeHist* treeh)
{
  return tree_ngram(treeh, 4, 1);
}

int
tree_noopenQr(TreeHist* treeh)
{
  InputTree* tree = treeh->tree;
  int pos = treeh->pos;
  int sz = tree->subTrees().size();
  InputTree  *subTree;
  InputTreesIter  subTreeIter = tree->subTrees().begin();
  int i = 0;
  bool sawOpen = false;
  
  for( ; ; subTreeIter++ )
    {
      if(i == pos) break; 
      subTree = *subTreeIter;
      assert(i < sz);
      const Term* trm = Term::get(subTree->term());
      if(trm->isOpen()) sawOpen=true;
      if(trm->isClosed()) sawOpen = false;
      i++;
    }
  if(sawOpen) return 0;
  else return 1;
}

int
tree_noopenQl(TreeHist* treeh)
{
  InputTree* tree = treeh->tree;
  int pos = treeh->pos;
  int hpos = treeh->hpos;
  InputTree  *subTree;
  InputTrees::reverse_iterator  subTreeIter = tree->subTrees().rbegin();
  int i = tree->subTrees().size()-1;
  bool sawOpen = false;
  bool sawClosed = false;
  
  for( ; ; subTreeIter++ )
    {
      if(i == pos) break;
      if(i > hpos) { i-- ; continue; }
      assert(i >= 0);
      subTree = *subTreeIter;
      const Term* trm = Term::get(subTree->term());

      if(trm->isClosed()) sawOpen = true;
      else if(trm->isOpen()) sawOpen = false;

      i--;
    }
  if(sawOpen) return 0;
  else return 1;
}

int
tree_B(TreeHist* treeh, int blInd)
{
  InputTree* tree = treeh->tree;
  int i;
  int pos = treeh->pos;
  int hpos = treeh->hpos;
  //cerr << "tb1 " << pos << " " << hpos << " " << *tree << endl;
  int sz = tree->subTrees().size();
  int wpos;
  assert(pos <= sz);
  //cerr << "tb " << pos << " " << hpos << " " << sz << endl;
  if(pos < 0) wpos = tree->start()-1;
  else if(sz == 0) wpos = tree->start()-1;
  else if(pos == sz) wpos = tree->finish();
  else
    {
      InputTreesIter iti = tree->subTrees().begin();
      i = 0;
      for( ; iti != tree->subTrees().end() ; iti++)
	{
	  if(i < pos) {i++; continue;}
	  InputTree* st = *iti;
	  if(pos < hpos) wpos = st->start()-1;
	  else if(pos > hpos) wpos = st->finish();
	  else if(blInd) wpos = st->start()-1;
	  else wpos = st->finish();
	  //cerr << "tbf " << *st << " " << wpos << endl;
	  break;
	}
    }
  //cerr << "tb2 " << wpos << endl;
  assert(wpos <= endPos);
  if(wpos < 0 || wpos == endPos) return Term::stopTerm->toInt();
  else return Term::get(sentence[wpos]->term())->toInt();
}

int
tree_Bl(TreeHist* treeh)
{
  return tree_B(treeh, 1);
}

int
tree_Br(TreeHist* treeh)
{
  return tree_B(treeh, 0);
}

InputTree*
tree_ruleTree(TreeHist* treeh, int ind)
{
  bool isM = false;
  CRuleBundle& crules = ClassRule::getCRules(treeh,ind);
  //cerr << "TRT " << crules.size() << endl;
  for(int i = 0 ; i < crules.size() ; i++)
    {
      InputTree* trdTree;
      trdTree = crules[i].apply(treeh);
      if(trdTree) return trdTree;
    }
  return NULL;
}

int
tree_ruleHead_third(TreeHist* treeh)
{
  InputTree* specTree = NULL;
  if(Feature::isLM) specTree = tree_ruleTree(treeh, 2);
  if(specTree) return headFromParTree(treeh->tree);
  InputTree* trdtree = tree_ruleTree(treeh,3);
  if(!trdtree) return nullWordInt;
  else return headFromTree(trdtree);
}

int
tree_watpos(int pos)
{
  if(pos < 0)
    {
      return nullWordInt;
    }
  ECString wrd = sentence[pos]->head();
  char tmp[512];
  ECString wrdl=toLower(wrd.c_str(), tmp);
  const WordInfo* wi = Pst::get(wrdl);
  assert(wi);
  int ans = wi->toInt();
  assert(ans >= 0);
  return ans;
}


int
tree_w1(TreeHist* treeh)
{
  int zpos = treeh->tree->start();
  //assert(treeh->tree->finish() == zpos);
  return tree_watpos(zpos-1);
}

int
tree_w2(TreeHist* treeh)
{
  int zpos = treeh->tree->start();
  //assert(treeh->tree->finish() == zpos);
  return tree_watpos(zpos-2);
}

void
addSubFeatureFns()
{
  /*
    0 t  tree_term
    1 l  tree_parent_term
    2 u  tree_pos
    3 h  tree_head
    4 i  tree_parent_head
    5 T  tree_true
    6 v  tree_parent_pos
    7 b  tree_term_before
    8 mE tree_mE
    //8 a  tree_term_after
    9 m  tree_grandparent_term
    10 w tree_grandparent_pos
    11 j tree_ruleHead_third  
    12 c tree_ccparent_term
    13 L1 tree_left1
    14 L2 tree_left2
    15 R1 tree_right1
    16 R2 tree_right2
    17 Qr tree_noopenQr
    18 L0 tree_left0;
    19 L3 tree_left3
    20 R3 tree_right3
    21 Ql tree_noopenQl
    22 Bl tree_Bl
    23 Br tree_Br
    24 vE tree_vE
    //25 E  tree_E
    25 w1 tree_w1
    26 w2 tree_w2
    */
  int (*funs[27])(TreeHist*)
    = {tree_term, tree_parent_term, tree_pos, tree_head, tree_parent_head,
       tree_true, tree_parent_pos, tree_term_before,
       tree_mE, tree_grandparent_term, tree_grandparent_pos,
       tree_ruleHead_third, tree_ccparent_term,
       tree_left1, tree_left2, tree_right1, tree_right2, tree_noopenQr,
       tree_left0, tree_left3, tree_right3,tree_noopenQl,tree_Bl,tree_Br,
       tree_vE, tree_w1, tree_w2 
    };
  int i;
  for(i = 0 ; i < 27 ; i++)
      SubFeature::Funs[i] = funs[i];
  string stopStr("STOP");
  const Term* stopTerm = Term::get(stopStr);
  stopTermInt = stopTerm->toInt();
  ECString wrdm = "^^";
  const WordInfo* wmi = Pst::get(wrdm);
  assert(wmi);
  int ans = wmi->toInt();
  nullWordInt = ans;
}
