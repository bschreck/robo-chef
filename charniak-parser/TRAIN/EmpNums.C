#include "EmpNums.h"
#include "utils.h"
#include <fstream>
#include <iostream>

int
whichEmpty(const ECString& emp)
{
  //return 1; // should make system not require empty type to be correct.
  if(emp == "0") return NULLEMP;
  if(emp == "*U*") return UNITEMP;
  int sz = emp.length();
  if(sz < 1)
    {
      return 0;
    }
  if(sz >= 5)
    {
      //ECString emp1(emp.substr(0, 5));
      ECString emp1(emp,0,5);
      if(emp1 == "*NOT*") return NOTEMP;
      if(emp1 == "*RNR*") return RNREMP;
      if(emp1 == "*ICH*") return ICHEMP;
      if(emp1 == "*EXP*") return EXPEMP;
      if(emp1 == "*PPA*") return PPAEMP;
    }
  if(sz >= 3)
    {
      //ECString emp1(emp.substr(0, 3));
      ECString emp1(emp,0, 3);
      if(emp1 == "*T*")
	{
	  return TREMP;
	}
      if(emp1 == "*NOT*") return NOTEMP;
      if(emp1 == "*?*") return QEMP; 
    }
  //ECString emp2(emp.substr(0,1));
  ECString emp2(emp,0,1);
  if(emp2 == "*") return NPEMP;
  return 0;
}

ECString
emptyFromInt(int emp)
{
  //return 1; // this should make system not require empty type to be correct.
  if(emp == NULLEMP) return "0";
  if(emp == UNITEMP) return "*U*";
  if(emp == NOTEMP) return "*NOT*";
  if(emp == RNREMP) return "*RNR*";
  if(emp == ICHEMP) return "*ICH*";
  if(emp == EXPEMP) return "*EXP*";
  if(emp == PPAEMP) return "*PPA*";
  if(emp == TREMP) return "*T*";
  if(emp == NOTEMP) return "*NOT*";
  if(emp == QEMP) return "*?*";
  if(emp == NPEMP) return "*";
  cerr << "Bad empty " << emp << endl;
  error("at loss");
  return "";
}

int
swEmpty(const ECString& wrd, const ECString& trm )
{
  if(wrd == "E_S") return 1;
  if(wrd == "[" && trm == "RM") return 1;
  if(wrd == "]" && trm == "RS") return 1;
  if(wrd == "+" && trm == "IP") return 1;
  return 0;
}
