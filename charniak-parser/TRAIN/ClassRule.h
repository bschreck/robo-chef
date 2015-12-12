
#ifndef CLASSRULE_H
#define CLASSRULE_H

#include "TreeHist.h"
#include <fstream>
#include <iostream>
#include <vector>
#include "InputTree.h"

#define MCALCRULES 10

/*
rules are of the form
A    B    C    D
which means "if you see an A under a B, apply function C,(1,2, or 3)
and if it returns sumthing of type D, use D's head as the third head
*/

class ClassRule
{
 public:
  ClassRule(int dd, int mm, int rr, int tt)
    : t_(tt), m_(mm), rel_(rr), d_(dd) {}
  ClassRule(const ClassRule& cr)
    : t_(cr.t()), m_(cr.m()), rel_(cr.rel()), d_(cr.d()) {}
  InputTree* apply(TreeHist* treeh);
  static void readCRules(ECString str);
  static vector<ClassRule>& getCRules(TreeHist* treeh, int wh);
  friend ostream& operator<<(ostream& os, const ClassRule& cr)
    {
      os << "{"<< cr.d() << "," << cr.m() << "," << cr.rel() << "," << cr.t() << "}";
      return os;
    }
  int d() const { return d_; }
  int m() const { return m_; }
  int t() const { return t_; }
  int rel() const { return rel_; }
 private:
  int d_;
  int m_;
  int t_;
  int rel_;
  static vector<ClassRule>  rBundles2_[100][50];
  static vector<ClassRule>  rBundles3_[100][50];
  static vector<ClassRule>  rBundlesm_[100][50];
};
    
typedef vector<ClassRule> CRuleBundle;

#endif /* ! CLASSRULE_H */
