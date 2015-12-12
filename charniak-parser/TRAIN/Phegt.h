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

#ifndef PHEGT_H
#define PHEGT_H

#include "ECString.h"

class Phegt
{
 public:
  Phegt() : t(-1), p(0){}
  Phegt(int t1, ECString& es) : t(t1), p(0)
    {
      e[0] = es[0];
      e[1] = es[1];
    }
  friend int operator== ( const Phegt& l, const Phegt& r)
    { return (l.greaterThan(r) == 0);}
  friend int operator> ( const Phegt& l, const Phegt& r)
    {return (l.greaterThan(r) > 0);}
  friend int operator< ( const Phegt& l, const Phegt& r)
    {return (l.greaterThan(r) < 0);}
  int t;
  char e[2];
  float p;
  int greaterThan(const Phegt& r) const;
  int greaterThan(int t, const char* e) const;
};


#endif /* ! PHEGT_H */
