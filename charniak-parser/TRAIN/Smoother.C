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

#include "Smoother.h"
#include "math.h"
#include <fstream>
#include <iostream>

float Smoother::bucketLims[14] =
  {0, .003, .01, .033, .09, .33, 1.01, 2.01, 5.1, 12, 30, 80, 200, 600};


int 
Smoother::
bucket(float val)
{
  for(int i = 0 ; i < 14 ; i++)
    if(val <= bucketLims[i])
      return i;
  return 14;
}



//JT new bucket taken from getProbs
int 
Smoother::
bucket(float val, int whichInt, int whichFt)
{
  float logFac = 1.0 / log(Feature::logFacs[whichInt][whichFt]);

  float lval = logFac *log(val);

  int lvi = (int)lval;
  lvi++;
  if(lvi <= 14) return lvi;
  return 14;
}


