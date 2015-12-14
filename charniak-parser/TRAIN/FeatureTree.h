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

#ifndef FEATURETREE_H
#define FEATURETREE_H

#include <map>
#include <assert.h>
#include <fstream>
#include <iostream>
#include "Feat.h"
#include <set>
#include "Feature.h"

class FeatureTree;
typedef map<int,FeatureTree*, less<int> > FTreeMap;
typedef map<int, int, less<int> > IntIntMap;
typedef map<int, Feat, less<int> > FeatMap;
typedef set<int, less<int> > IntSet;
typedef map<int,IntSet, less<int> > IntSetMap;


class FeatureTree
{
public:
  FeatureTree() : specFeatures(0), auxNd(NULL), back(NULL), marked(-1),
    count(0), featureInt(-1) {}
  FeatureTree(int i) : ind(i), specFeatures(0), auxNd(NULL), back(NULL),
     marked(-1), count(0), featureInt(-1) {}
  FeatureTree(int i, FeatureTree* b)
    : ind(i), specFeatures(0), auxNd(NULL), back(b), marked(-1),
      count(0), featureInt (-1) {}
  FeatureTree(istream& is);
  int  readOneLevel0(istream& is);
  FeatureTree* next(int val, int auxCnt);
  FeatureTree* follow(int val, int auxCnt);
  static FeatureTree* roots(int which) { return roots_[which]; }
  static FeatureTree*& root() { return roots_[Feature::whichInt]; }
  void   printFTree(int asVal, ostream& os);
  friend ostream&  operator<<(ostream& os, const FeatureTree& ft);
  int ind;
  int count;
  int featureInt;
  int specFeatures;
  float marked;
  FeatureTree* back;
  FeatureTree* auxNd;
  FeatMap feats;
  //BinaryArray featA;
  FTreeMap subtree;
  static int       totParams;
  static int       minCount;
 private:
  static FeatureTree* roots_[15];
  void read(istream& is, FTypeTree* ftt);
  void othReadFeatureTree(istream& is, FTypeTree* ftt, int c);
  void         printFfCounts(int asVal, int depth,
			     ostream& os, FTypeTree* ftt);
  void printFfCounts2(int asVal, int depth, ostream& os, FTypeTree* ftt);
};

#endif /* ! FEATURETREE_H */
