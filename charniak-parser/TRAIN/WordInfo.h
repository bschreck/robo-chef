/*
 * Copyright 1997, Brown University, Providence, RI.
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

#ifndef WORDINFO_H
#define WORDINFO_H

#include "ECString.h"

struct Phsgt
{
  int   term;           // indicates term 
  int   classNum;
  float prob;           // for term prob
};

class WordInfo
{
  friend class Pst;
 public:
  WordInfo() : lexeme_(), st_( 0 ), c_( 0 ), stSize_(0) { }
  int          stSize() const { return stSize_; }
  int          c() const { return c_; }
  const ECString&  lexeme() const { return lexeme_; }
  int          toInt() const { return n_; }
  // protected:
  //const Phsgt& subTerms() const { return *st_; }  //???;
  ECString  lexeme_;
  int       stSize_;
  int       c_;
  int       n_;
  Phsgt*    st_;
};

#endif /* ! WORDINFO_H */
