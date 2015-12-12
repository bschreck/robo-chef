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

#include <math.h>
#include <fstream>
#include <sys/resource.h>
#include <iostream>
#include <unistd.h>
#include "ECArgs.h"
#include "ECString.h"
#include "utils.h"
#include "FeatIter.h"
#include "Feature.h"
#include "Term.h"

#define NUMPASSES 4
#define NUMNEWTPASSES 4

int whichInt;

FeatureTree* features;

ECString conditionedType;
int passN;
int hic = 0;

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
  while(parft->ind == AUXIND) parft = parft->back;
  //if(badPreReqs(ft,parft)) return NULL;
  FeatMap::iterator fmi = parft->feats.find(f->ind());
  if(fmi == parft->feats.end()) return NULL;
  return &((*fmi).second);
}

void
initFeatVals()
{
  FeatIter fi(features); 
  int m, i;
  Feat* f;
  for( ; fi.alive() ; fi.next() )
    {
      f = fi.curr;
      int fhij = f->cnt();
      assert(fhij > 0);
      int hij = f->toTree()->count;
      assert(hij > 0);
      assert(hij >= fhij);
      Feat* fj = parentFeat(f);
      int hj, fhj;
      if(fj)
	{
	  hj = fj->toTree()->count;
	  fhj = fj->cnt();
	}
      else
	{
	  fhj = 1;
	  hj = 1; // this sets val to fhij/hij;
	  //hj=FeatureTree::totCaboveMin[whichInt][Feature::assumedFeatVal]+1;
	}
      assert(hj > 0);
      assert(fhj > 0);
      assert(hj >= fhj);
      //float val = (float)(fhij * hj)/(float)(fhj * hij);
      float val = ((float)fhij/(float)hij);
      fi.curr->g() = val;
      //cerr << *(f->toTree()) << " " << f->ind()
	//   << " " << val << endl;
      if(!(val > 0))
	{
	  cerr << fhij << " " << hj << " " << fhj << " " << hij << endl;
	  assert(val > 0);
	}
    }
}

int
main(int argc, char *argv[])
{
   struct rlimit 	core_limits;
   core_limits.rlim_cur = 0;
   core_limits.rlim_max = 0;
   setrlimit( RLIMIT_CORE, &core_limits );

   ECArgs args( argc, argv );
   Feat::Usage = ISCALE;
   ECString  path( args.arg( 1 ) );
   Term::init(path);

   conditionedType = args.arg(0);
   cerr << "start iScale: " << conditionedType << endl;
   //Pst pst(path);

   Feature::init(path, conditionedType);
   whichInt = Feature::whichInt;
   ECString fHp(path);
   fHp += conditionedType;
   fHp += ".f";
   ifstream fHps(fHp.c_str());
   if(!fHps)
     {
       cerr << "Could not find " << fHp << endl;
       assert(fHps);
     }

   features = new FeatureTree(fHps);

   //features->subtree;
   
   initFeatVals();
   
   ECString gt(path);
   gt += conditionedType;
   gt += ".g";
   ofstream gtstream(gt.c_str());
   assert(gtstream);
   gtstream.precision(3);

   FTreeMap::iterator ftmi = features->subtree.begin();
   for( ; ftmi != features->subtree.end() ; ftmi++)
     {
       int afv = (*ftmi).first;
       (*ftmi).second->printFTree(afv, gtstream);
     }
   return 1;
}
