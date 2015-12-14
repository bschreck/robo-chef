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

#include <ctype.h>
#include <fstream>
#include <iostream>
#include <assert.h>
#include "InputTree.h"
#include "headFinder.h"
#include "utils.h"
#include "Term.h"
#include "EmpNums.h"
#include "auxify.h"
#include "ECString.h"

int              InputTree::pageWidth = 75;  //used for prettyPrinting
void    breakString(ECString str, ECString& part1, char& brk, ECString& part2);

int
okFTag(ECString nc)
{
  static ECString ftgs[22] = {"TMP", "LOC", "ADV", "TPC", "BNF", "DIR",
			    "EXT", "NOM", "DTV", "LGS", "PRD", "PUT",
			    "SBJ", "VOC", "MNR", "PRP", "CLR", "CLF",
			    "HLN", "TTL", "DEI", "PLE"};
  int i;
  for(i = 0 ; i < 22 ; i++) if(nc == ftgs[i]) return 1;
  return 0;
}


InputTree::
~InputTree()
{
  InputTree  *subTree;
  InputTreesIter  subTreeIter = subTrees_.begin();
  for( ; subTreeIter != subTrees_.end() ; subTreeIter++ )
    {
      subTree = *subTreeIter;
      delete subTree;
    }
}

InputTree::
InputTree(istream& is) : parent_(NULL)
{
  readParse(is);
}


istream&
operator >>( istream& is, InputTree& parse )
{
  if(parse.word() != "" || parse.term() != ""
     || parse.subTrees().size() > 0)
    error( "Reading into non-empty parse." );
  parse.readParse(is);
  return is;
}

void
InputTree::
flushConstit(istream& is)
{
  int totalOpen = 1;
  for( ; is ; )
    {
      char ch;
      is.get(ch);
      if(ch == '(') totalOpen++;
      else if(ch == ')')
	{
	  totalOpen--;
	  if(totalOpen == 0) return;
	}
    }
}



void
InputTree::
readParse(istream& is)
{
  int pos = 0;
  start_ = pos;
  finish_ = pos;

  ECString temp = readNext(is);
  if(!is) return;
  if(temp != "(") error("Should have seen an open paren here.");
  /* get annotated symbols like NP-OBJ.  term_ = NP ntInfo_ = OBJ */
  temp = readNext(is);
  term_ = "S1";
  if(temp != "(")
    {
      if(temp == "S1" || temp == "TOP")
	{
	  temp = readNext(is);
	}
      else
	{
	  cerr << temp << " is not legal topmost type" << endl;
	  flushConstit(is);
	  return;
	}
    }
  if(temp != "(") error("Should have seen second open paren here.");
  for (;;)
    {
      InputTree* nextTree = newParse(is, pos, this);
      subTrees_.push_back(nextTree);
      finish_ = pos;

      headTree_ = nextTree->headTree_;
      temp = readNext(is);
      if (temp==")") break;
      if (temp!="(")
	{
	  cerr << *this << endl;
	  error("Should have open or closed paren here.");
	}
    }
  int hpos=headPosFromTree(this);
  headTree()=ithInputTree(hpos,subTrees_)->headTree();
}

InputTree*
InputTree::
fixNPBifNecessary(InputTree* nextTree, ECString trm)
{
  if(nextTree->term() != "NPB") return nextTree;
  if(trm == "NP") return nextTree;
  //at this point we have an NPB under a non-NP;
  InputTrees subTrs;
  subTrs.push_front(nextTree);
  InputTree* ans = new InputTree(nextTree->start(), nextTree->finish(),
				 "", "NP", "", subTrs,
				 NULL, NULL);
  nextTree->parentSet() = ans;
  ans->headTree() = nextTree->headTree();
  return ans;
}


InputTree*
InputTree::
newParse(istream& is, int& strt, InputTree* par)
{
  int strt1 = strt;
  ECString wrd;
  ECString trm;
  ECString ntInf;
  ECString ftag1 = "";
  ECString ftag2 = "";
  ECString nstring;
  InputTrees subTrs;

  parseTerm(is, trm, ntInf, ftag1, ftag2, nstring);
  //cerr << "APT " << trm << " " << ntInf << endl;
  InputTree* lstTree = NULL;
  for( ; ; )
    {
      ECString temp = readNext(is);
      if(temp == "(")
	{
	  //cerr << "Bef recnp" << endl;
	  InputTree* nextTree = newParse(is, strt, NULL);
	  //cerr << "Ret with " << *nextTree << endl;
	  if(!nextTree) continue;
	  lstTree = nextTree;
	  subTrs.push_back(nextTree);
	}
      else if(temp == ")") break;
      else
	{
	      if(trm != "-NONE-" && trm != "-DFL-")
		{
		  //cerr << "strt= " << strt << " " << temp << endl;
		  strt++;
		  wrd = temp;
		}
	}
    }
  if(Term::Language == "Ch" && trm == "PU") trm = wrd;
  if (!Term::get(trm))
    {
      cerr<<trm<<endl;
      assert(Term::get(trm));
    }
  if(wrd == "" && subTrs.size() == 0) return NULL;
  InputTreesIter iti = subTrs.begin();
  if(trm == "S" || trm == "NP")
    {
      bool sawNP = false;
      iti = subTrs.begin();
      for( ; iti != subTrs.end() ; iti++)
	{
	  InputTree* subt = (*iti);
	  if(subt->term() != "NP" && subt->term() != "NPB") continue;
	  if(subt->hTag() != "POS" || trm == "S")
	    {
	      sawNP=true;
	      break;
	    }
	}
      //if(!sawNP && trm == "S") trm = "SG";
      //else if(!sawNP && trm == "NP") trm = "NPB";
    }
  /* fixes bugs in Chinese Treebank */
  if(Term::Language == "Ch")
    {
      if(wrd!="" && !(Term::get(trm))->terminal_p())
	{
	  cout<<trm<<wrd<<" changed to NN"<<endl;
	  trm="NN";
	}
      if(wrd=="" && Term::get(trm)->terminal_p() )
	{
	  cout<<trm<<" changed to NP"<<endl;
	  trm="NP";
	}
    }
  InputTree* ans = new InputTree(strt1, strt, wrd, trm, ntInf, subTrs,
				 par, NULL);
  ans->fTag() = ftag1;
  ans->fTag2() = ftag2;
  ans->num() = nstring;
  //cerr << "First ans " << *ans << endl;

  iti = subTrs.begin();
  for(; iti != subTrs.end() ; iti++)
    {
      InputTree* st = *iti;
      st->parentSet() = ans;
    }
  if(wrd != "" || trm == "-NONE-")
    {
      //ans->term() = auxify(wrd,trm);
      ans->headTree() = ans;
      assert( (Term::get(ans->term()))->terminal_p() );
    }
  else
    {
      int hpos = headPosFromTree(ans);
      //cerr << "RR " << hpos << endl;
      ans->headTree() = ithInputTree(hpos, subTrs)->headTree();
      assert( (Term::get(ans->headTree()->term()))->terminal_p() );
    }
  return ans;
}

ECString&
InputTree::
readNext( istream& is )
{
  static ECString word[1024];
  static int num = 0;

  // if we already have a word, use it, and increment pointer to next word;
  if( word[num] != "" )
    {
      return word[num++];
    }
  int tempnum;
  for(tempnum = 0 ; word[tempnum] != "" ; tempnum++ )
    word[tempnum] = "";
  num = 0;
  // then go though next input, separating "()[]";
  int    wordnum  = 0 ;
  ECString  temp;
  is >> temp;
  for( tempnum = 0 ; tempnum < temp.length() ; tempnum++ )
    {
      char tempC = temp[tempnum];
      if(tempC == '(' || tempC == ')')
	 //|| tempC == '[' || tempC == ']' )
	{
	  if( word[wordnum] != "" )
	    wordnum++;
	  word[wordnum++] += tempC;
	}
      else word[wordnum] += temp[tempnum];
    }
  return word[num++];
}

/* if we see NP-OBJ make NP a, and -OBJ b */
void
InputTree::
parseTerm(istream& is, ECString& a, ECString& b, ECString& f1, ECString& f2, ECString& n)
{
  ECString temp1 = readNext(is);
  if(temp1 == "(" || temp1 == ")") error("Saw paren rather than term");

  ECString temp;
  int len = temp1.length() -1;
  assert(len >= 0);
  if(temp1[0] == '^')  //Remove ^ from pos;
    //temp = temp1.substr(1, temp1.length()-1);
    {
      ECString temp3(temp1, 1, temp1.length()-1);
      temp = temp3;
    }
  else temp = temp1;

  ECString p1, p2;
  char br = 'a';
  breakString(temp, p1, br, p2);
  a = p1;
  if(br == 'a')
    {
      b = "";
      return;
    }
  b = br;
  b += p2;
  n = numSuffex(b);
  if(br == '|') return;
  if(p2 == n) return;
  if(br != '#')
    {
      br = 'a';
      ECString q1, q2;
      breakString(p2, q1, br, q2);
      if(q1 == n) return;
      f1 = q1;
      if(br == 'a') return;
      if(br != '#')
	{
	  ECString r1, r2;
	  br = 'a';
	  breakString(q2, r1, br, r2);
	  f2 = r1;
	  if(br != '#') return;
	}
    }
}


ostream&
operator <<( ostream& os, const InputTree& parse )
{
  parse.prettyPrint( os, 0, false );
  return os;
}

void
InputTree::
printproper( ostream& os ,bool withhead) const
{
  if( word_.length() != 0 )
    {
      if (withhead){
	    os << "(" << term_ << ntInfo_<< neInfo_ <<'['<<(headTree_->word())<<']'<<" " << word_ << ")";
      }else{
	    os << "(" << term_ << ntInfo_<< neInfo_ <<" " << word_ << ")";
      }
    }
  else
    {
      os << "(";
      os <<  term_ << ntInfo_ << neInfo_;
      if (withhead) os<<"["<<(headTree_->word())<<"]";
      ConstInputTreesIter  subTreeIter= subTrees_.begin();
      InputTree  *subTree;
      for( ; subTreeIter != subTrees_.end() ; subTreeIter++ )
	{
	  subTree = *subTreeIter;
	  os << " ";
	  subTree->printproper( os,withhead );
	}
      os << ")";
    }
}

void InputTree::
prettyPrintWithHead(ostream& os) const
{
  prettyPrint(os,0,false,true);
}

void
InputTree::
prettyPrint(ostream& os, int start, bool startingLine, bool withhead) const
{
  if(start >= pageWidth) //if we indent to much, just give up and print it.
    {
      printproper(os,withhead);
      return;
    }
  if(startingLine)
    {
      os << "\n";
      int numtabs = start/8;
      int numspace = start%8;
      int i;
      for( i = 0 ; i < numtabs ; i++ ) os << "\t"; //indent;
      for( i = 0 ; i < numspace ; i++ ) os << " "; //indent;
    }
  /* if there is enough space to print the rest of the tree, do so */
  if(spaceNeeded() <= pageWidth-start || word_ != "")
    {
      printproper(os,withhead);
    }
  else
    {
      os << "(";
      if (withhead){
	  os << term_ << ntInfo_ << neInfo_<<'['<<(headTree_->word())<<"]";
      }else{
	  os << term_ << ntInfo_ << neInfo_;
      }
      os << " ";
      /* we need 2 extra spaces, for "(", " "  */
      int newStart = start + 2 + term_.length() + ntInfo_.length()
	             + neInfo_.length();
      //move start to right of space after term_ for next thing to print
      start++; //but for rest just move one space in.
      ConstInputTreesIter  subTreeIter = subTrees_.begin();
      InputTree  *subTree;
      int whichSubTree = 0;
      for( ; subTreeIter != subTrees_.end() ; subTreeIter++ )
	{
	  subTree = *subTreeIter;
	  if(whichSubTree++ == 0)
	    {
	      subTree->prettyPrint(os, newStart, false,withhead);
	    }
	  else
	    {
	      subTree->prettyPrint(os, start, true,withhead);
	    }
	}
      os << ")";
    }
}

/* estimates how much space we need to print the rest of the currently
   print tree */
int
InputTree::
spaceNeeded() const
{
  int needed = 1; // 1 for blank;
  int wordLen = word_.length();
  needed += wordLen;
  needed += 3; //3 for () and " ";
  needed += term_.length();
  needed += ntInfo_.length();
  needed += neInfo_.length();
  if(word_ != "") return needed;
  ConstInputTreesIter  subTreeIter = subTrees_.begin();
  InputTree  *subTree;
  for( ; subTreeIter != subTrees_.end() ; subTreeIter++ )
    {
      subTree = *subTreeIter;
      needed += subTree->spaceNeeded();
    }
  return needed;
}

void
InputTree::
make(EcSPairs& wordTermList)
{
  if(word_ != "")
    {
      EcSPair wtp(term_, word_);
      wordTermList.push_back(wtp);
    }
  else
    {
      ConstInputTreesIter subTreeIter = subTrees().begin();
      InputTree  *subTree;
      for(; subTreeIter != subTrees().end() ; subTreeIter++)
	{
	  subTree = *subTreeIter;
	  subTree->make(wordTermList);
	}
    }
}

InputTree*
ithInputTree(int i, const list<InputTree*> l)
{
  if(i >= l.size()) return NULL;
  list<InputTree*>::const_iterator li = l.begin();
  for(int j = 0 ; j < i ; j++)
    {
      assert(li != l.end());
      li++;
    }
  return *li;
}

int
InputTree::
isEmpty()
{
  if(term_ == "-NONE-") return whichEmpty(word_);
  const Term* trm = Term::get(term_);
  int trmInt = trm->toInt();
  if(trm->terminal_p()) return 0;

  ConstInputTreesIter subTreeIter = subTrees().begin();
  InputTree  *subTree;
  int numNone = 0;
  for(; subTreeIter != subTrees().end() ; subTreeIter++)
    {
      subTree = *subTreeIter;
      if(!subTree->isEmpty()) return 0;
      int we = whichEmpty(subTree->word());
      numNone++;
    }
  if(numNone == 0) return 0;
  if(term_ == "SBAR" && numNone == 2)
    {
      return SBAREMP;
    }
  else if(numNone == 1)
    {
      int we = whichEmpty(subTrees().front()->word());
      return we;
    }
  else if(numNone == 2) return OTHEMP;
  else
    {
      cerr << "was not expecting this tree" << endl;
      cerr << *this << endl;
      return ERROREMP;
    }
}

int
InputTree::
isUnaryEmpty()
{
  if(term_ == "-NONE-") return 0;
  const Term* trm = Term::get(term_);
  int trmInt = trm->toInt();
  if(trm->terminal_p()) return 0;

  ConstInputTreesIter subTreeIter = subTrees().begin();
  InputTree  *subTree;
  int numNone = 0;
  int numUnary = 0;
  for(; subTreeIter != subTrees().end() ; subTreeIter++)
    {
      subTree = *subTreeIter;
      if(!subTree->isEmpty())
	{
	  if(subTree->term() != term_)  return 0;
	  else numUnary++;
	  if(numUnary > 1) return 0;
	}
      else numNone++;
    }
  if(numNone == 1 && numUnary >0 )
    {
      assert(numUnary == 1);
      return 1;
    }
  else return 0;
}

ECString
numSuffex(ECString str)
{
  int sz = str.length();
  if(sz <= 1) return "";
  int i;
  ECString ans;
  for(i = sz-1 ; i >= 0 ; i--)
    {
      char c = str[i];
      if(!isdigit(c)) break;
      ans = c + ans;
    }
  return ans;
}

void
breakString(ECString str, ECString& part1, char& brk, ECString& part2)
{
  size_t sz = str.length();
  if(sz <= 1)
    {
      part1 = str;
      return;
    }
  size_t pose = str.find("|");
  if(pose >= sz) pose = str.find("^");
  //^VP^RP payoff found in sw2067.mrg, this strips second "^" ???;
  if(pose < sz)
    {
      brk = '|';
      assert(pose >= 0);
      //part1 = str.substr(0, pose);
      ECString temp1(str,0,pose);
      part1 = temp1;
      int len2 = sz - pose -1;
      assert(len2 >= 0);
      //part2 = str.substr(pose+1,len2);
      ECString temp2(str,pose+1,len2);
      part2 = temp2;
      return;
    }
  size_t pos = str.find("-");
  size_t pos2 = str.find("#");
  size_t pos3 = str.find("=");
  size_t spos = pos < pos2 ? pos : pos2;
  spos = spos < pos3 ? spos : pos3;
  if(spos >= sz || spos == 0)
    {
      part1 = str;
      return;
    }
  if(spos == pos || spos == pos3) brk = '-';
  else brk = '#';
  assert(spos >= 0);
  //part1 =str.substr(0, spos);
  ECString temp3(str,0, spos);
  part1 = temp3;
  int len3 = sz-spos-1;
  if(len3 < 0)
    {
      cerr << sz << " " << spos << " " << str << endl;
      assert(len3 >= 0);
    }
  //part2 = str.substr(spos+1,len3);
  ECString temp4(str,spos+1,len3);
  part2 = temp4;
}



bool
InputTree::
isCodeTree()
{
  if(subTrees().size() == 0) return false;
  InputTree* subt = subTrees().front();
  if(subt->term() == "CODE") return true;
  return false;
}

bool
InputTree::
readCW(istream& is)
{
  char temp[256];
  int tot = 0;
  int totstar = 0;
  for( ; ; )
    {
      if(!is)
	{
	  if(tot < 2) return false;
	  error("Ran out of data before copyright ended?");
	}
      is >> temp;
      tot++;
      if(temp[0] == '*') totstar++;
      if(totstar == 20) break;
      if(tot > 100) error("Copyright too long?");
    }
  return true;
}
