/*
 * Copyright 2005 Brown University, Providence, RI.
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

class AnsStrCounts
{
 public:
  AnsStrCounts()
  {
    for(int i = 0 ; i < 11 ; i++)
      numWords[i] = numCounts[i] = numSents[i] = 0;
  }
  void updateCounts(int len)
  {
    int buk = len/10;
    numWords[buk] += len;
    numCounts[buk] += AnsTreeStr::numCreated;
    numSents[buk]++;
    AnsTreeStr::numCreated = 0;
  }
  int numWords[11];
  int numCounts[11];
  int numSents[11];
  void showCounts();
};

void
AnsStrCounts::
showCounts()
{
  for(int i = 0 ; i < 11 ; i++)
    {
      cerr << i << "\t";
      if(numWords[i] != 0)
	{
	  float cps = (float)numCounts[i]/(float)numSents[i];
	  float wps = (float)numWords[i]/(float)numSents[i];
	  cerr << numSents[i] << "\t"<< (float)numCounts[i]
	       << "\t" << wps
	       << "\t" << cps
	       << "\t" << 10.0*wps*sqrt(wps);
	}
      cerr << endl;
    }
}
