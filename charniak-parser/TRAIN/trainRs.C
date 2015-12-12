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
#include"trainRsUtils.h"
extern int pass;
extern int whichInt;
extern int sentenceCount;
extern ECString conditionedType;
extern FeatureTree* tRoot;
extern bool procGSwitch;
extern StringSet wordSet;


int
main(int argc, char *argv[])
{

   struct rlimit 	core_limits;
   core_limits.rlim_cur = 0;
   core_limits.rlim_max = 0;
   setrlimit( RLIMIT_CORE, &core_limits );

   ECArgs args( argc, argv );
   assert(args.nargs() == 2);
   conditionedType = args.arg(0);
   cerr << "start trainRs: " << conditionedType << endl;
  
   ECString  path( args.arg( 1 ) );
   if(args.isset('M')) Feature::setLM();
   if(args.isset('L')) Term::Language = args.value('L');

   Term::init(path);

   readHeadInfo(path);
   Pst pst(path);
   if(Feature::isLM) ClassRule::readCRules(path);

   addSubFeatureFns();
   Feature::init(path, conditionedType); 

   whichInt = Feature::whichInt;
   int ceFunInt = Feature::conditionedFeatureInt[Feature::whichInt];
   Feature::conditionedEvent
     = SubFeature::Funs[ceFunInt];

   Feat::Usage = PARSE;
   ECString ftstr(path);
   ftstr += conditionedType;
   ftstr += ".g";
   ifstream fts(ftstr.c_str());
   if(!fts)
     {
       cerr << "Could not find " << ftstr << endl;
       assert(fts);
     }
   tRoot = new FeatureTree(fts); //puts it in root;

   cout.precision(3);
   cerr.precision(3);

   lamInit();

   InputTree* trainingData[1001];
   int usedCount = 0;
   sentenceCount = 0;
   for( ;  ; sentenceCount++)
     {
       if(sentenceCount%10000 == 1)
	 {
	   // cerr << conditionedType << ".tr "
	   //<< sentenceCount << endl;
	 }
       if(usedCount >= 1000) break;
       InputTree*     correct = new InputTree;  
       cin >> (*correct);

       if(correct->length() == 0) break;
       if(!cin) break;
       EcSPairs wtList;
       correct->make(wtList); 
       InputTree* par;
       par = correct;
       trainingData[usedCount++] = par;
     }
   if(Feature::isLM) pickLogBases(trainingData,sentenceCount);
   procGSwitch = true;
   for(pass = 0 ; pass < 10 ; pass++)
     {
       if(pass%2 == 1) cout << "Pass " << pass << endl;
       goThroughSents(trainingData, sentenceCount);
       updateLambdas();
       //printLambdas(cout);
       zeroData();
     }
   ECString resS(path);
   resS += conditionedType;
   resS += ".lambdas";
   ofstream res(resS.c_str());
   res.precision(3);
   printLambdas(res);
   printLambdas(cout);
   cout << "Total params = " << FeatureTree::totParams << endl;
   cout << "Done: " << (int)sbrk(0) << endl;

}
