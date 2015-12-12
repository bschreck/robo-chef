#ifndef GETPROBS_H
#define GETPROBS_H

#include <fstream>
#include <sys/resource.h>
#include <iostream>
#include <unistd.h>
#include <set>
#include <math.h>
#include "ClassRule.h"
#include "ECArgs.h"
#include "Feature.h"
#include "FeatureTree.h"
#include "InputTree.h"
#include "headFinder.h"
#include "treeHistSf.h"
#include "Pst.h"
#include "Smoother.h"
#include "TreeHist.h"
#include "Term.h"

#include "trainRsUtils.h"


class getProbs{

 public:
  static void init(ECString path);

};
float getProb(InputTree* tree, int pos, int whichInt);
void initGetProb(string path);
 

#endif
