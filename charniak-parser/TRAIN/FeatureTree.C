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

#include "FeatureTree.h"
#include "ECString.h"
#include <set>
#include "Feature.h"
#include "math.h"

int FeatureTree::totParams = 0;
FeatureTree* FeatureTree::roots_[15];
int FeatureTree::minCount = 1;

FeatureTree*
FeatureTree::
next(int val, int auxCnt)
{
  if(!auxCnt)
    {
      FeatureTree* ans = subtree[val];
      if(ans) return ans;
      ans = new FeatureTree(val,this);
      subtree[val] = ans;
      return ans;
    }
  if(!auxNd) auxNd = new FeatureTree(AUXIND,this);
  return auxNd->next(val, auxCnt-1);
}

FeatureTree* 
FeatureTree::
follow(int val, int auxCnt)
{
  if(!auxCnt)
    {
      FTreeMap::iterator fti = subtree.find(val);
      if(fti == subtree.end()) return NULL;
      return (*fti).second;
    }
  //assert(auxNd); //???;
  if(!auxNd) return NULL;
  return auxNd->follow(val, auxCnt-1);
}

/* basic format
   assumedNum //e.g., 55 (np)
        rule# count
        ...
	--
	48 //this would be np under adjp
             rule# count
	     ...
        --  //end  of rules
        -- //end of 48
	49
	...
	--  //end of 49
	...
	--  // end of g(rtl)s
   Pringint is brown into 2 procedures.  The first takes a position
   in the tree and determines if the data there supports it (i.e.,
   there is enough for any one conditioned value).  if so, it
   prints out the link value, and the conditioned values. It then
   calls f2 on itself.
*/

FeatureTree::
FeatureTree(istream& is)
  : specFeatures(0), auxNd(NULL), back(NULL), ind(ROOTIND), marked(-1)
{
  int done = 0;
  while(is && !done)
    {
      int val = readOneLevel0(is);
      if(val == -1) done = 1;
    }
  roots_[Feature::whichInt] = this;
}
 
int
FeatureTree::
readOneLevel0(istream& is)
{
  int nextInd;
  ECString nextIndStr;
  is >> nextIndStr;
  //cerr << "nIS " << nextIndStr << endl;
  if(!is) return -1;
  if(nextIndStr == "Selected") return -1;
  nextInd = atoi(nextIndStr.c_str());
  FeatureTree* nft = new FeatureTree(nextInd);
  nft->read(is,Feature::ftTree[Feature::whichInt].left);
  subtree[nextInd] = nft;
  nft->back = this; 
  return nextInd;
}

void
FeatureTree::
read(istream& is, FTypeTree* ftt)
{
  ECString indStr;
  int indI;
  is >> count;
  int cf, cs, c;
  is >> cf;  
  is >> cs;
  //cerr << "r " << count << " " << ftt->n << " " << cf << endl;
  for(c=0 ; c < cf ; c++)
    {
      is >> indI;
      Feat& nf = feats[indI];
      nf.ind() = indI;
      nf.ft_ = this;
      if(Feat::Usage != PARSE)
	{
	  int v;
	  is >> v;
	  nf.cnt() = v;
	  //cerr << indI << "\t" << v << endl;
	}
      else
	{
	  float v;
	  is >> v;
	  nf.g() = v;
	  //cerr << indI << "\t" << v << endl;
	}
    }
  othReadFeatureTree(is,ftt,cs);
}

void
FeatureTree::
othReadFeatureTree(istream& is, FTypeTree* ftt,int cs)
{
  featureInt = ftt->n;
  //cerr<< "fi " << featureInt << " " << cs << endl;
  if(featureInt <= 0 && featureInt != AUXIND)
    {
      assert(featureInt > 0);
    }
  ECString indStr;
  int indI, c;
  for(c = 0 ; c < cs ; c++)
    {
      is >> indI;
      FeatureTree* ntr = subtree[indI];
      if(!ntr)
	{
	  ntr = new FeatureTree(indI, this);
	  subtree[indI] = ntr;
	}
      assert(ftt->left);
      ntr->read(is, ftt->left);
    }
  if(!ftt->right) return;
  assert(!auxNd);
  is >> indStr;
  if(indStr != "A")
    {
      cerr << "fi = " << featureInt << " " << cs << " " << indStr << endl;
      assert(indStr == "A");
    }
  int ac;
  is >> ac;  
  //cerr << "An " << ac << endl;
  auxNd = new FeatureTree(AUXIND,this);
  auxNd->othReadFeatureTree(is, ftt->right, ac);
}


/* basic format
   assumedNum //e.g., 55 (np)
        rule# count
        ...
	--
	48 //this would be np under adjp
             rule# count
	     ...
	     --
	49
	...
	--  //end of 49
	...
	--  // end of g(rtl)s
	A  //indicates that there is another group of features here.
	2  //np's with head's pos->toInt() == 2;
   Pringint is brown into 2 procedures.  The first takes a position
   in the tree and determines if the data there supports it (i.e.,
   there is enough for any one conditioned value).  if so, it
   prints out the link value, and the conditioned values. It then
   calls f2 on itself.
*/

int
markedSize(FTreeMap& sub)
{
  if(Feat::Usage != SEL) return sub.size();
  int i = 0;
  FTreeMap::iterator subFeatIter = sub.begin();
  FeatureTree* subT;
  for( ; subFeatIter != sub.end() ; subFeatIter++ )
    {
      subT = (*subFeatIter).second;
      if(subT->marked >= 0) i++;
    }
  return i;
}

void
FeatureTree::
printFTree(int asVal, ostream& os)
{
  printFfCounts(asVal, 0, os, Feature::ftTree[Feature::whichInt].left);
}

void
FeatureTree::
printFfCounts(int asVal, int depth, ostream& os, FTypeTree* ftt)
{
  //cerr << "pff " << " " << depth << " " << count << endl;
  FeatMap::iterator conditionedIter = feats.begin();
  int i;
  //if(count < minCount) return;
  if(Feat::Usage == SEL && marked < 0) return;
  for(i = 0 ; i < depth ; i++)
    os << "\t";
  os << ind << "\t" ;
  os << count;
  os << "\t" << feats.size();
  os << "\t" << markedSize(subtree);
  os << "\n"; 

  /* now print out the counts for the features */
  conditionedIter = feats.begin();
        
  for(; conditionedIter != feats.end() ; conditionedIter++)
    {
      Feat* ft = &((*conditionedIter).second);
      int cnt = ft->cnt();
      int fval = (*conditionedIter).first;
      for(i = 0 ; i <= depth ; i++)
	os << "\t";
      os << fval << "\t" ;
      // during iterative scaling we print out feature tree to look;
      // at gammas;
      if(Feat::Usage == ISCALE || Feat::Usage == KNCOUNTS)
	{
	  float gval = (*conditionedIter).second.g();
	  os << gval;
	}
      else
	os << cnt;
      os << endl;
      totParams++;
    }
  //for(i = 0 ; i <= depth ; i++) os << "\t";
  //os << "--\n"; //This indicates end of feature/count pairs;
  os << "\n";
  printFfCounts2(asVal, depth, os, ftt);
}

void
FeatureTree::
printFfCounts2(int asVal, int depth, ostream& os, FTypeTree* ftt)
{
  //cerr << "pff " << asVal << endl;
  FTreeMap::iterator subFeatIter = subtree.begin();
  FeatureTree* subT;
  for( ; subFeatIter != subtree.end() ; subFeatIter++ )
    {
      subT = (*subFeatIter).second;
      subT->printFfCounts(asVal, depth+1, os, ftt->left);
    }
  if(ftt->right)
    {
      for(int i = 0 ; i < depth ; i++)
	os << "\t";
      os << "A\t";
      if(!auxNd)
	{
	  os << "0\n";
	  return;
	}
      os << markedSize(auxNd->subtree);
      os << "\n";
      //cerr << "Ap " << auxNd->subtree.size() << endl;
      auxNd->printFfCounts2(asVal, depth, os, ftt->right); //???;
    }
}
  
ostream&
operator<< ( ostream& os, const FeatureTree& t )
{
  int temp[10];
  int i = 0;
  const FeatureTree* ft = &t;
  for( ; ft ; ft = ft->back )
    if(ft->ind >= -1) temp[i++] = ft->ind;
  os << "<";
  for(int j = i-1 ; j >= 0 ; j--)
    {
      os  << temp[j] ;
      if(j > 0) os << ",";
    }
  os << ">";
  return os;
}

      
    


