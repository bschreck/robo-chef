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

#include <time.h>
#include <sys/resource.h>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <math.h>
#include "Field.h"
#include "GotIter.h"
#include "Wrd.h"
#include "InputTree.h"
#include "Bchart.h"
#include "ECArgs.h"
#include "MeChart.h"
#include "headFinder.h"
#include "Params.h"
#include "AnsHeap.h"
#include "TimeIt.h"
#include "extraMain.h"
#include "Link.h"

MeChart* curChart;
Params 		    params;

int
main(int argc, char *argv[])
{
   ECArgs args( argc, argv );
   //AnsTreeHeap::print = true;
   /* o = basic, but not debugging, output.
      l = length of sentence to be proceeds 0-40 is default
      n = work on each #'th line.
      d = print out debugging info at level #
      t = report timings (requires o)
   */

   //cerr << "Starting wwBCTest" << endl;
   params.init( args );
   if( args.nargs() > 2 || args.nargs() < 2 )	// require path name 
     error( "Need exactly two args." );
   ECString  path( args.arg( 0 ) );
   generalInit(path);
   bool histPoints[500];
   for(int i = 0 ; i < 500 ; i++) histPoints[i] = false;
   histPoints[0] = true;
   if(Bchart::Nth == 50)
     histPoints[1] = histPoints[9] = histPoints[24] = histPoints[49] = true;
   ParseStats  totPst[Bchart::Nth];
   TimeIt timeIt;

   ECString testSString = args.arg(1);
   ifstream testSStream(testSString.c_str());
   if( !testSStream )
     {
       cerr << "Could not find " << testSString << endl;
       error( "No testSstream" );
     }

    ECString      pstatStreamName( params.fileString());
    pstatStreamName  += "PStatInfo/pStat";
    pstatStreamName += params.numString();
    pstatStreamName += ".txt";
    ofstream    pstatStream( pstatStreamName.c_str(), ios::out);
    if( !pstatStream )
      {
	cerr << "Looking to output to " << pstatStreamName << endl;
	error( "unable to open pstat stream" );
      }

   int      sentenceCount = 0;  //counts all sentences so we can use 1/50;

   double  totGram  = 0.0;
   double  totMix  = 0.0;
   double  totTri  = 0.0;
   int  totWords = 0;
   int totSents = 0;
   int totUnparsed = 0;
   double log600 = log2(600.0);
   for( ; testSStream ; totSents++)
     {
       InputTree     correct;  
       InputTree*    cuse;
       testSStream >> correct;
       cuse = &correct;
       if( !testSStream ) break;
       int len = correct.length();
       if(len > params.maxSentLen) continue;
       //cerr << "Len = " << len << endl;
       if( !params.field().in(sentenceCount) )
	 {
	   sentenceCount++;
	   continue;
	 }
       if(sentenceCount < -1)
	 {
	   sentenceCount++;
	   continue;
	 }
       //cerr << correct << endl;
       //cerr << sentenceCount << endl;
       //if(sentenceCount > 0) break;
       sentenceCount++;
       list<ECString>  wtList;
       correct.make(wtList);
       vector<ECString> poslist;
       correct.makePosList(poslist);
       SentRep sr( wtList );  // used in precision calc

       if(args.isset('t')) timeIt.befSent();

       InputTree::setEquivInts(poslist);
       MeChart*	chart = new MeChart( sr );
       curChart = chart;
       if(args.isset('t') ) timeIt.lastTime = clock();
       
       double tmpCrossEnt = chart->parse( );
       Item* topS = chart->topS();
       if(!topS)
	 {
	   totUnparsed++;
	   cerr << "Parse failed" << endl;
	   cerr << correct << endl;
	   error(" could not parse "); 
	   delete chart;
	   continue;
	 }
       
       // compute the outside probabilities on the items so that we can
       // skip doing detailed computations on the really bad ones 
       if(args.isset('t') ) timeIt.betweenSent(chart);

       chart->set_Alphas();

       Bst& bst = chart->findMapParse();
       if( bst.empty()) error( "mapProbs did not return answer" );
       float bestF = -1;
       ParseStats bestPs;
       int i;
       vector<InputTree*> saveTrees;
       saveTrees.reserve(Bchart::Nth);
       vector<double> saveProbs;
       saveProbs.reserve(Bchart::Nth);
       int numVersions = 0;
       Link diffs(0);
       int numDiff = 0;
       //cerr << "Need num diff: " << Bchart::Nth << endl;
       for(numVersions = 0 ; ; numVersions++)
	 {
	   short pos = 0;
	   Val* val = bst.next(numVersions);
           if(!val)
	     {
	       //cerr << "Breaking" << endl;
	       break;
	     }
           InputTree*  mapparse = inputTreeFromBsts(val,pos,sr);
	   bool isU;
	   diffs.is_unique(mapparse, isU);
	   // cerr << "V " << isU << " " << numVersions << *mapparse << endl;
	   if(isU)
	     {
	       saveTrees.push_back(mapparse);
	       numDiff++;
	     }
	   else
	     {
	       delete mapparse;
	     }
	   if(numDiff >= Bchart::Nth) break;
	   if(numVersions > 20000) break;
	 }
       pstatStream << totSents << "\t" << numDiff << "\n";
       pstatStream << correct << "\n";
       for(i = 0 ; i < numDiff ; i++)
	 {
           InputTree*  mapparse = saveTrees[i];
	   InputTree::trips.clear();
	   double logP =log2(bst.nth(i)->prob());
	   logP -= (sr.length()*log600);
	   pstatStream <<  logP << "\n";
	   if(Bchart::prettyPrint) pstatStream << *mapparse << "\n\n";
	   else
	     {
	       mapparse->printproper(pstatStream);
	       pstatStream << "\n";
	     }
	   InputTree::trips.clear();
	   ParseStats pSt;
	   cuse->recordGold( pSt );
	   mapparse->precisionRecall(pSt);
	   float newF = pSt.fMeasure();
	   if(newF > bestF)
	     {
	       bestF = newF;
	       bestPs = pSt;
	     }
	   if(histPoints[i])
	     {
	       totPst[i] += bestPs;
	     }
	   delete mapparse;
	 }
       if(numDiff < Bchart::Nth)
	 {
	   for(i = numDiff ; i < Bchart::Nth ; i++)
	     {
	       if(histPoints[i]) totPst[i] += bestPs;
	     }
	 }
       pstatStream << endl;
       //aSC.updateCounts(correct.length());
       if(Feature::isLM)
	 {
	   double lgram = log2(bst.sum());
	   lgram -= (sr.length()*log600);
	   double pgram = pow(2,lgram);
	   totGram -= lgram;
	   totWords += sr.length()+1;
	   double iptri = .00001;
	   iptri =chart->triGram();;
	   double ltri = (log2(iptri)-sr.length()*log600);
	   totTri -= ltri;
	   double ptri = pow(2.0,ltri);
	   double pcomb1 = (0.667 * pgram)+(0.333 * ptri);
	   double pcomb2 = (0.8 * pgram)+(0.2 * ptri);
	   double lcom1 = log2(pcomb1);
	   double lcom2 = log2(pcomb2);
	   totMix -= lcom1;
	   cerr << totSents << "\t";
	   cerr << pow(2.0,totGram/(double)totWords);
	   cerr <<"\t" <<  pow(2.0,totTri/(double)totWords);
	   cerr << "\t" << pow(2.0,totMix/(double)(totWords));
	   cerr << endl;
	 }
       if(totSents%50 == 1)
	 {
	   cerr << totSents << "\t";
	   for(int i = 0 ; i < Bchart::Nth ; i++)
	     if(histPoints[i])
	       {
		 cerr << i << " " << totPst[i].fMeasure() << "\t";
	       }
	   cerr << endl;
	 }
       if(args.isset('t') ) timeIt.aftSent();

       delete chart;
     }
   for(int i = 0 ; i < Bchart::Nth ; i++)
     if(histPoints[i])
       {
	 cerr << i << " " << totPst[i].fMeasure() << "\t";
       }
   cerr << endl;
   if(Feature::isLM)
     {
       cerr << pow(2.0,totGram/(double)totWords);
       cerr <<"\t" <<  pow(2.0,totTri/(double)totWords);
       cerr << "\t" << pow(2.0,totMix/(double)(totWords));
       cerr << endl;
     }
   if( args.isset('t') ) timeIt.finish(totSents);

   return 0;
}
