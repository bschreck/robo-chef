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

#ifndef INPUTTREE_H
#define INPUTTREE_H

#include <list>
#include "ECString.h"
#include "utils.h"

class InputTree;
typedef list<InputTree*> InputTrees;
typedef InputTrees::iterator InputTreesIter;
typedef InputTrees::const_iterator ConstInputTreesIter;
typedef pair<ECString,ECString> EcSPair;
typedef list<EcSPair> EcSPairs;
typedef EcSPairs::iterator EcSPairsIter;

bool scorePunctuation( const ECString trmString );

class  InputTree
{
 public:
  InputTree(const InputTree& p);
  InputTree(istream& is);
  InputTree() : start_(0), finish_(0), parent_(NULL) {}
  InputTree(int s, int f, const ECString w, const ECString t, const ECString n,
	    InputTrees& subT, InputTree* par, InputTree* headTr)
    : start_(s), finish_(f), word_(w), term_(t), ntInfo_(n),
      subTrees_(subT), parent_(par), headTree_(headTr),
      num_(""), traceTree_(NULL){}
  ~InputTree();

  friend istream& operator >>( istream& is, InputTree& parse );
  friend ostream& operator <<( ostream& os, const InputTree& parse );
  int         start() const { return start_; }
  int         length() const { return (finish() - start()); }
  int         finish() const { return finish_; }
  const ECString word() const { return word_; }  
  ECString&      word() { return word_; }  
  const ECString term() const { return term_; }
  ECString&      term() { return term_; }
  const ECString ntInfo() const { return ntInfo_; }
  ECString&      ntInfo() { return ntInfo_; }
  const ECString fTag() const { return fTag_; }
  ECString&      fTag() { return fTag_; }
  const ECString fTag2() const { return fTag2_; }
  ECString&      fTag2() { return fTag2_; }
  const ECString head() { return headTree_->word(); }
  const ECString hTag() { return headTree_->term(); }
  ECString       neInfo() const { return neInfo_; }
  ECString&      neInfo() { return neInfo_; }
  InputTrees& subTrees() { return subTrees_; }
  InputTree*& headTree() { return headTree_; }
  InputTree*& traceTree() { return traceTree_; }
  InputTree*  traceTree() const { return traceTree_; }
  InputTree*  parent() { return parent_; }
  InputTree*&  parentSet() { return parent_; }
  ECString       num() const { return num_; }
  ECString&      num() { return num_; }
  int          isEmpty();
  int          isUnaryEmpty();
  void        make(EcSPairs& str);
  bool        isCodeTree();
  void        readParse(istream& is);
  static bool readCW(istream& is);
  bool        ccTree();
  bool        ccChild();
  static int  pageWidth;     
  void prettyPrintWithHead(ostream& os) const;
 protected:
  InputTree*     newParse(istream& is, int& strt, InputTree* par);
  ECString&  readNext( istream& is );
  void        parseTerm(istream& is, ECString& a, ECString& b, ECString& c,
			ECString& f2, ECString& n);
  void        printproper( ostream& os, bool withhead=false ) const;
  void        prettyPrint(ostream& os, int start, bool startLine, bool withhead=false) const;
  int         spaceNeeded() const;
  void        flushConstit(istream& is);
  InputTree*  fixNPBifNecessary(InputTree* nextTree, ECString trm);
  
  InputTree*  parent_;
  int         start_;
  int         finish_;
  ECString   word_;
  ECString   term_;
  ECString   fTag_;
  ECString   fTag2_;
  ECString   ntInfo_;
  ECString      num_;
  ECString   neInfo_;
  InputTree*  traceTree_;
  InputTree*  headTree_;
  InputTrees  subTrees_;
};

InputTree* ithInputTree(int i, const list<InputTree*> l);
ECString     numSuffex(ECString str);
int        okFTag(ECString nc);

#endif /* ! INPUTTREE_H */
