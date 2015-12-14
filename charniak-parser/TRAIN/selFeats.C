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
#include <unistd.h>
#include <iostream>
#include "Feat.h"
#include "FeatIter.h"
#include "math.h"
#include "utils.h"
#include "ECArgs.h"
#include <map>

multimap<float, FeatureTree*, less<float> > featRank;

float percentDesiredFeatures = 0;
int totStates = 0;
int totSelectedStates = 0;
int totDesiredFeatures = 0;
int whichInt;

float logLcomp(int fhi, int hi, int fhj, int hj);
float logLikely(FeatureTree* f);
void markFeat(FeatureTree*& f, float fr);
Feat* parentFeat(Feat* f);
FeatureTree* parentTree(FeatureTree* ft);
void markFeats();

float
logLcomp(int fhi, int hi, int fhj, int hj)
{
  if(fhi == 0)
    {
      return 0.0;
    }
  double lfhi = log((float)fhi);
  double lhi = log((float)hi);
  double lfhj = log((float)fhj);
  double lhj = log((float)hj);
  double ans = fhi * (lfhi - lhi - lfhj + lhj);
  //cerr << "logLcomp(" << fhi << ", " << hi << ", " << fhj
	// << ", " << hj << ") = "
	  //<< fhi << " * (" << lfhi << " - " << lhi << " - "
	 // << lfhj << " + " << lhj << ") = " << ans << endl;
  return (float)ans;
}

float
logLikely(FeatureTree* ft)
{
  //cerr << "LL " << *ft << endl;
  int ftc = ft->count;
  assert(ftc > 0);
  FeatureTree* ftp = parentTree(ft);
  assert(ftp);
  int ftpc = ftp->count;
  assert(ftpc > 0);
  FeatMap::iterator fmi = ft->feats.begin();
  float ans = 0;
  /* we compute the sum for each feature f on ft of
     |f|(log(p(f|ft))-log(p(f|ftb)))  = logLcomp(fc, ftc, fpc, ftpc) */
  
  for( ; fmi != ft->feats.end() ; fmi++)
    {
      Feat* f = &((*fmi).second);
      Feat* fp = parentFeat(f);
      assert(fp);
      int fc = f->cnt();
      int fpc = fp->cnt();
      float val = logLcomp(fc, ftc, fpc, ftpc);
      //cerr << "lcomp " << *f << " = " << val << endl;
      ans += val;
    }
  return ans;
}
/*
float
logLikely(FeatureTree* f)
{
  int fc = f->cnt();
  assert(fc > 0);
  int hij = f->toTree()->count;
  assert(hij > 0);
  assert(hij >= fhij);
  Feat* fj = parentFeat(f);
  assert(fj);
  int hj = fj->toTree()->count;
  assert(hj > 0);
  int fhj = fj->cnt();
  assert(fhj > 0);
  assert(hj >= fhj);
  int fhnij = fhj - fhij;
  if(!(fhnij >= 0))
    {
      cerr << fhnij << " not >= 0 for " << *f << " " << *fj << endl;
      cerr << *(f->toTree()) << endl;
      cerr << *(fj->toTree()) << endl;
      assert(fhnij >= 0);
    }
  int hnij = hj - hij;
  assert(hnij >= 0);
  if(!(hnij >= fhnij))
    {
      cerr << hnij << " " << fhnij << " for " << *f << endl;
      assert(hnij >= fhnij);
    }
  int nfhj = hj - fhj;
  int nfhij = hij - fhij;
  assert(nfhj >= 0);
  assert(nfhij >= 0);
  if(!(hij >= nfhij))
    {
      cerr << nfhij << " " << hij << " for " << *f << endl;
      assert(hij >= nfhij);
    }
  int nfhnij = hnij - fhnij;
  assert(nfhnij >= 0);
  assert(hnij >= nfhnij);
  float ans;
  ans = logLcomp(fhij, hij, fhj, hj);
  ans += logLcomp(fhnij, hnij, fhj, hj);
  ans += logLcomp(nfhij, hij, nfhj, hj);
  ans += logLcomp(nfhnij, hnij, nfhj, hj);
  if(ans < -0.001)
    {
      cerr << ans << "\t" << *(f->toTree()) 
	   << f->ind() << endl;
      cerr << fhij << " " << hij << ",  " << fhj << " " << hj << ",  " << fhnij
	<< " " << hnij << ",  " << nfhij << "  " << nfhj << " " << nfhnij << endl;
      assert(ans >= -0.001);
    }
  return ans;
}
*/
bool
badPreReqs(FeatureTree* ft, FeatureTree* parft)
{
  int fInt = ft->featureInt;
  //cerr << "ft to feat gives " << *ft << " " << fInt << endl;
  int cprf = Feature::fromInt(fInt, whichInt)->condPR;
  int pfInt = parft->featureInt;
  //cerr << "ft to feat gives " << *parft << " " << pfInt << endl;
  int cprpf = Feature::fromInt(pfInt, whichInt)->condPR;
  if(cprpf < 0) return false;
  if(cprpf == cprf) return false;
  //cerr << "badPr for " << *ft << " " << *parft << " " << fInt << " "
    //<< pfInt << endl;
  return true;
}

Feat*
parentFeat(Feat* f)
{
  FeatureTree* ft = f->toTree();
  assert(ft);
  FeatureTree* parft = ft->back;
  if(!parft) return NULL; 
  if(parft->ind == ROOTIND) return NULL;
  while(parft->ind == AUXIND)
    {
      FeatureTree* parft2 = parft->back;
      assert(parft2);
      parft = parft2;
    }
  if(f->ind() == -1 && badPreReqs(ft,parft)) return NULL;
  FeatMap::iterator fmi = parft->feats.find(f->ind());
  if(fmi == parft->feats.end()) return NULL;
  return &((*fmi).second);
}

FeatureTree*
parentTree(FeatureTree* ft)
{
  assert(ft);
  FeatureTree* parft = ft->back;
  if(!parft) return NULL; 
  if(parft->ind == ROOTIND) return parft;
  while(parft->ind == AUXIND)
    {
      FeatureTree* parft2 = parft->back;
      assert(parft2);
      parft = parft2;
    }
  return parft;
}

void
doRanking()
{
  FeatureTree* root = FeatureTree::roots(whichInt);
  assert(root);
  FeatTreeIter fi(root);
  FeatureTree* f;
  for( ; fi.alive() ; fi.next() )
    {
      f = fi.curr;
      //cerr << "Looking at " << *f << endl;
      /* features at level 1 are automatically used */
      FeatureTree* parFt = parentTree(f);
      assert(parFt);
      assert(parFt->ind != AUXIND);
      if(parFt->ind == ROOTIND) markFeat(f, 1);
      else
	{
	  float ll = logLikely(f);
	  pair<const float, FeatureTree*> ipair(ll, f);
	  if(ll > 0) featRank.insert(ipair);
	}
      /*
	int cnt = f->cnt();
	if(cnt >= 20) markFeat(f, 1);
	else if(!parFt) markFeat(f, 1);
	else
	{
	  float ll = logLikely(f);
	  if(cnt >= 10 && ll > 0.1) markFeat(f,1);
	  else if(cnt >= 5 && ll > 2.0) markFeat(f,1);
	  else if(cnt >= 3 && ll > 8.0) markFeat(f, 1);
	}
	*/

      totStates++;
    }
}

void
markFeats()
{
  multimap<float, FeatureTree*, less<float> >::reverse_iterator
    fRankIter = featRank.rbegin();
  int i = 0 ;
  bool justCut = false;
  for( ; fRankIter != featRank.rend() ; fRankIter++)
    {
      float fRank = (*fRankIter).first;
      FeatureTree* ft = (*fRankIter).second;
      /* if(i++%500 == 2) cerr << i << " " << fRank << " " << *ft << endl;*/
      //if(ft->featureInt == 4)
      //cerr << i << " " << fRank << " " << *ft << endl;

      if(totSelectedStates < totDesiredFeatures)
	{
	  markFeat(ft, fRank);
	}
      else if(!justCut)
	{
	  justCut = true;
	  cerr << "Just cut off at " << i << " " << fRank << endl;
	}
    }
}

void
markFeat(FeatureTree*& f, float fr)
{
  //assert(!(f->marked == 0)); 
  if(f->marked > -1) return; 
  f->marked = fr;
  if(f->ind != AUXIND) totSelectedStates++;
  FeatureTree* pf = f->back;
  if(!pf) return;
  markFeat(pf,fr);
}


int
main(int argc, char *argv[])
{
  struct rlimit 	core_limits;
  core_limits.rlim_cur = 0;
  core_limits.rlim_max = 0;
  setrlimit( RLIMIT_CORE, &core_limits );

  ECArgs args( argc, argv );
  assert(args.nargs() == 3);
  ECString conditionedType = args.arg(0);
  Feat::Usage = SEL;

  percentDesiredFeatures = (float)atoi(args.arg(1).c_str())/100.0;
  cerr << "start selFeats: " << conditionedType
       << " " << percentDesiredFeatures  << endl;
  ECString  path( args.arg( 2 ) );
  ECString fHp(path);
  fHp += conditionedType;
  fHp += ".ff";
  Feature::init(path, conditionedType);
  ifstream fHps(fHp.c_str());
  new FeatureTree(fHps);

  whichInt = Feature::whichInt;

  cerr << "Before doRanking" << endl;
  doRanking();
  totDesiredFeatures = (int) (percentDesiredFeatures * totStates);
  cerr << "Before markFeats" << endl;
  markFeats();

  ECString resS(path);
  resS += conditionedType;
  resS += ".f";
  ofstream res(resS.c_str());

  FeatureTree* root = FeatureTree::roots(whichInt);
  FTreeMap::iterator ftmIter = root->subtree.begin();
  cerr << "About to print featuretree" << endl;
  for( ; ftmIter != root->subtree.end() ; ftmIter++)
    {
      int asVal = (*ftmIter).first;
      FeatureTree* subRoot = root->subtree[asVal];
      subRoot->printFTree(asVal, res);
    }
  res << "\n\nSelected " << totSelectedStates << " of "
       << totStates << endl;
  return 1;
}
