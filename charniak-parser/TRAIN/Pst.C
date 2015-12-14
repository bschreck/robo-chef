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
#include "Pst.h"
#include "Phegt.h"
#include "Term.h"
#include "utils.h"

WordInfo* Pst::h_ = NULL;
int       Pst::hSize_ = 0;
int       Pst::numClasses_[50]; //unused;
int       Pst::classOffsets_[50];//unused;

Pst::
Pst(ECString& path)
{
  readPhsgt(path);
}

const Phsgt*
Pst::
useSubTermC(const ECString& head, const int trm )const
{
  const WordInfo* uh = useHeadC( head );
  if( !uh ) return NULL;
  int  sz = uh->stSize();
  for( int i = 0 ; i < sz ; i ++ )
    {
      const Phsgt* ut = &(uh->st_[i]);
      if( ut->term == trm )
	{
	  return ut;
	}
    }
  return NULL;
}

const WordInfo*
Pst::
useHeadC(const ECString& head)
{
  int top = hSize();
  int bot = -1;
  for( ; ; )
    {
      if( top <= bot+1 )return NULL;
      int mid = (top+bot)/2;
      const WordInfo* midH = &(h_[mid]);
      ECString midString = midH->lexeme();
      if( head == midString )
	return midH;
      else if( head < midString )
	top = mid;
      else bot = mid;
    }
}

list<double>
Pst::
wordPlistConstruct(const ECString& head, int word_num)
{
  list<double> ans;
  char temp[512];
  ECString headL(toLower(head.c_str(), temp));
  const WordInfo* wi = useHeadC( headL );
  if( wi )
    {
      int  sz = wi->stSize();   
      for( int i = 0 ; i < sz ; i ++ )
	{
	  Phsgt& wti = wi->st_[i];
	  int    tInt = wti.term;
	  if(tInt > Term::lastTagInt()) continue;
	  double prob = psktt(head,tInt,word_num);
	  ans.push_back(tInt);
	  ans.push_back(prob);
	  if(prob == 0)
	    cerr << "Warning, prob = 0 for word = " << head
	      << " and pos = " << tInt << endl;
	  //cerr << "wordPlist: " << word << "\t" << tInt
	    // << "\t" << prob << endl;
	}
    }
  else
    {
      for(int i = 0 ; i <= Term::lastTagInt() ; i++)
	{
	  double phut = pHugt(i);
	  if(phut == 0) continue;
	  double prob = psutt(head,i,word_num);
	  ans.push_back(i);
	  ans.push_back(prob);
	}
    }
  return ans;
}

/* computs p(s|t) for terminals by finding out if it is known, and
   calling psktt for known words, and psutt for unknown words */
   
double
Pst::
pstt(ECString& shU, int t, int word_num)
{
  char temp[512];
  ECString sh(toLower(shU.c_str(), temp));
  const Term* tTerm = Term::fromInt(t);
  double phst = pHst(sh, t);
  double ans;
  if(phst > 0)
    ans =  psktt(shU, t, word_num);
  else ans = psutt(shU, t, word_num);
  return ans;
}

double
Pst::
psktt(const ECString& shU, int t, int word_num)
{
  char temp[512];
  ECString sh(toLower(shU.c_str(), temp));
  double ans = pHst(sh, t);
  double phcp = pCapgt(shU,t, word_num);
  ans *= phcp;
  double put = pHugt(t);
  ans *= (1-put);
  //cerr << "psktt( " << shU << " | " << t << " ) = " << ans << endl;
  return ans;
}
  
double
Pst::
psutt(const ECString& shU, int t, int word_num)
{
  //cerr << "Unknown word: " << shU << " for tag: " << t << endl; 
  double ans = pHugt(t);
  //cerr << "pHugt = " << ans << endl;
  if(ans == 0) return 0;
  double phyp = pHypgt(shU,t);
  ans *= phyp;
  //cerr << "pHypgt = " << phyp << endl;
  double phcp = pCapgt(shU,t, word_num);
  ans *= phcp;
  ans *= .000001;
  if(Term::fromInt(t)->openClass())
    {
      char temp[512];
      ECString sh(toLower(shU.c_str(),temp));
      float phegt = pegt(sh,t);
      if(phegt == 0) phegt = .00001;
      //if(phegt == 0) phegt = .00005;
      //cerr << "pegt( " << sh << " | " << t << " ) = " << phegt << endl;
      ans *= phegt;
    }
  else
    ans *= .00000001;

  //cerr << "psutt( " << shU << " | " << t << " ) = " << ans << endl;
  return ans;
}

double
Pst::
pCapgt(const ECString& shU, int t, int word_num)
{
  if(word_num == 0) return 1;
  //cerr << "pCapgt = " << pcap << endl;
  if(shU.length() < 2) return 1;  //ignore words of length 1;
  char temp[512];
  ECString sh(toLower(shU.c_str(),temp));
  bool cap = false;
  if(shU[0] != sh[0] && shU[1] == sh[1]) cap = true;
  double pcap = pHcapgt(t);  
  return cap ? pcap : (1 - pcap);
}

double
Pst::
pHypgt(const ECString& shU, int t)
{
  bool hyp = false;
  if( shU.find("-") >= 0) hyp = true;
  double phyp = pHhypgt(t);  
  return hyp ? phyp : (1 - phyp);
}
  
float
Pst::
pegt(ECString& sh, int t)
{
  int len = sh.length();
  if(len < 3) return .01;
  ECString e = sh.substr(len -2, 2);
  float phegt = pHegt(e,t);
  return phegt;
}

float
Pst::
pHegt(ECString& es, int t)
{
  char e[2];
  e[0] = es[0];
  e[1] = es[1];
  
  int top = egtSize_;
  int bot = -1;
  for( ; ; )
    {
      if( top <= bot+1 )
	return 0.0;
      int mid = (top+bot)/2;
      Phegt& midH = pHegt_[mid]; 
      
      int gt =  midH.greaterThan(t, e);
      if(gt  == 0) return midH.p;
      else if( gt == 1 ) top = mid;
      else bot = mid;
    }
}
  
void
Pst::
readTermProbs(ECString& pTString, ECString& pUstring)
{
  ifstream pTstream(pTString.c_str());
  assert(pTstream);
  ignoreComment(pTstream);
  ifstream pUstream(pUstring.c_str());
  assert(pUstream);
  ignoreComment(pUstream);
  int i, j;
  for( i = 0 ; i <=  Term::lastNTInt() ; i++ )
    {
      int t;
      pUstream >> t;
      float p;
      pUstream >> p;
      pHugt(t) = p;
      pUstream >> p;
      if(p == 0) p = .00001;  //Anything might be capitalized;
      pHcapgt(t) = p;
      pUstream >> p;
      pHhypgt(t) = p;
    }
  int numpT;
  pTstream >> numpT;
  pHegt_ = new Phegt[numpT];
  egtSize_ = numpT;

  i = 0;
  while(pTstream)
    {
      int t;
      char e0;
      char e1;
      float p;
      pTstream >> t;
      if(!pTstream) break;
      assert(i < numpT);
      pTstream >> e0;
      pTstream >> e1;
      pTstream >> p;
      pHegt_[i].t = t;
      pHegt_[i].e[0] = e0;
      pHegt_[i].e[1] = e1;
      pHegt_[i].p = p;
      i++;
    }
}

void
Pst::
readPhsgt(ECString& path)
{
  ECString pstString(path);
  pstString += "pSgT.txt";
  ifstream pstStream(pstString.c_str());
  if(!pstStream)
    {
      cerr << "Could not find " << pstString << endl;
      assert(pstStream);
    }
  ignoreComment(pstStream);

  int numWords;
  pstStream >> numWords;
  hSize_ = numWords;
  h_ = new WordInfo[numWords];
  for(int i = 0 ; i < numWords ; i++)
    {
      ECString wrd;
      pstStream >> wrd;
      assert(pstStream);
      
      int ts[80];
      float ps[80];

      WordInfo& wi = h_[i];
      wi.n_ = i;
      wi.lexeme_ = wrd;
      int j;
      for(j = 0 ; ; j++)
	{
	  assert(j < 80);
	  ECString temp;
	  pstStream >> temp;
	  if(temp == "|")
	    {
	      pstStream >> wi.c_;
	      break;
	    }
	  ts[j] = atoi(temp.c_str());
	  pstStream >> ps[j];
	}
      wi.stSize_ = j;
      wi.st_ = new Phsgt[j];
      for(int k = 0 ; k < j ; k++)
	{
	  wi.st_[k].term = ts[k];
	  wi.st_[k].prob = ps[k];
	  wi.st_[k].classNum = 0;
	}
    }
  assert(pstStream);
  ECString temp;
  pstStream >> temp;
  if(pstStream)
    {
      cerr << "Junk still left in pHsgt.txt " << temp << endl;
      assert(!pstStream);
    }
}
      
	  
      
const ECString&
Pst::
fromInt(int i)
{
  assert(i < hSize_);
  return h_[i].lexeme();
}

void
Pst::
classReader(const ECString& path)
{
  ECString clpath(path);
  clpath += "classes/";
  ECString ncs(clpath);
  ncs += "numClasses.txt";
  ifstream ncss(ncs.c_str());
  assert(ncss);
  int classOffset = 0;
  while(ncss)
     {
       int posNum;
       ncss >> posNum;
       if(!ncss) break;
       ECString posName;
       ncss >> posName;
       int numCs;
       ncss >> numCs;
       numClasses_[posNum] = numCs;
       classOffsets_[posNum] = classOffset;
       if(numCs > 1)
	 {
	   /*
	   for(int i = 0 ; i < numCs ; i++)
	     {
	       int clfinal = i + classOffset;
	       cerr << "Cdef " << posNum <<"\t"<<i<< "\t" << clfinal << endl;
	     }
	     */
	   classOffset += numCs;
	 }
     }
  clpath += "classdata/cALL.txt";
  ifstream css(clpath.c_str());
  assert(css);
  while(css)
    {
      int pos;
      css >> pos;
      if(!css) break;
      for( ; ; )
	{
	  ECString wnumStr;
	  css >> wnumStr;
	  if(wnumStr == "---")
	    {
	      for ( ; ; )
		{
		  ECString temp;
		  css >> temp;
		  if(temp == "---") break;
		}
	      break;
	    }
	  int wnum = atoi(wnumStr.c_str());
	  int cl;
	  css >> cl;
	  int clfinal = cl+classOffsets_[pos];
	  classNum(wnum,pos) = clfinal; 
	}
      classOffset += numClasses_[pos];
    }
}

int&
Pst::
classNum(int wnum, int pos)
{
  WordInfo& wi = h_[wnum];
  int  sz = wi.stSize();
  for( int i = 0 ; i < sz ; i ++ )
    {
      Phsgt* ut = &(wi.st_[i]);
      if( ut->term == pos )
	{
	  return ut->classNum;
	}
    }
  error("should not be here");
}

int
Pst::
classNm(int wnum, int pos)
{
  WordInfo& wi = h_[wnum];
  int  sz = wi.stSize();
  for( int i = 0 ; i < sz ; i ++ )
    {
      Phsgt* ut = &(wi.st_[i]);
      if( ut->term == pos )
	{
	  return ut->classNum;
	}
    }
  return -1;  
}
