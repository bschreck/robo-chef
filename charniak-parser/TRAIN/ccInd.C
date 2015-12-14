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

/* this can be copied over to nparser05/TRAIN */
#include "ccInd.h"
#include "InputTree.h"
#include "Term.h"
#include "Feature.h"

int
ccIndFromTree(InputTree* tree)
{
  InputTreesIter  subTreeIter = tree->subTrees().begin();
  ECString trmNm = tree->term();
  bool sawComma = false;
  bool sawColen = false;
  bool sawCC = false;
  bool sawOTHNT = false;
  int numTrm = 0;
  int pos = 0;
  const Term* trm = Term::get(trmNm);
  int tint = trm->toInt();
  /*Change next line to indicate which non-terminals get specially
    marked to indicate that they are conjoined together */
  if(!trm->isNP() && !trm->isS() && !trm->isVP()) return tint;
  for( ; subTreeIter != tree->subTrees().end() ; subTreeIter++ )
    {
      InputTree* subTree = *subTreeIter;
      ECString strmNm = subTree->term();
      const Term* strm = Term::get(strmNm);
      if(pos != 0 && strm->isCC()) sawCC = true;
      else if(strmNm == trmNm) numTrm++;
      else if(pos != 0 && strm->isComma()) sawComma = true;
      else if(pos != 0 && strm->isColon()) sawColen = true;
      else if(!strm->terminal_p()) sawOTHNT = true;
      pos++;
    }
  if(trmNm == "NP" && numTrm == 2 && !sawCC) return Term::lastNTInt()+1;
  if((sawComma || sawColen || sawCC) && numTrm >= 2) return tint+Term::lastNTInt();
  return tint;
}
