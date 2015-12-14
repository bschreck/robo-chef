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

   params.init( args );
   if( args.nargs() > 2 || args.nargs() < 2 )	// require path name 
     error( "Need exactly two args." );
   ECString  path( args.arg( 0 ) );
   generalInit(path);

   TimeIt timeIt;


   int      sentenceCount = 0;  //counts all sentences so we can use 1/50;
   ECString dataFilesFile = args.arg(1);
   ECString dataFilesPrefix = args.arg(2);
   ifstream dataFiles(dataFilesFile.c_str());
   if(!dataFiles)
     {
       cerr << "Could not find data file: " << dataFilesFile << endl;
       return 0;
     }
   for( ; ; )
     {
       ECString datastring = dataFilesPrefix;
       ECString file;
       dataFiles >> file;
       if(!dataFiles) break;
       datastring += file;
       ifstream ifs(datastring.c_str());
       if(!ifs)
	 {
	   cerr << "Could not find: " << datastring << endl;
	   continue;
	 }

       for(; ;)
	 {
	   cerr << "BB" << endl;
	   if(!ifs) break;
	   SentRep sr;  //dummy;
	   cerr << "AA" << endl;
	   MeChart*	chart = new MeChart(sr);
	   cerr << "DD" << endl;
	   bool cont = true;
	   InputTree parse(ifs);
	   if(!ifs) break;
	   cerr << parse << endl;
	   chart->edgesFromTree(&parse);

	   cerr << "readEdgets" << endl;
	   chart->set_Betas();
	   cerr << "set Betas" << endl;
	   Item* s1 = chart->topS();
	   if(!s1) break;
	   chart->set_Alphas();

	   AnsTreeStr& att = chart->findMapParse();
	   if(att.numCreated < 1) error("mapProbs did not return answer");
	   AnsTree* at = &att.trees[0];
	   short num = 0;
	   InputTree*  mapparse = inputTreeFromAnsTree(at,num,sr);
	   if(args.isset('o'))
	     {
	       cout << *mapparse << endl; //???;
	       cout << "Prob = " << att.probs[0] << endl;
	     }
	   delete chart;
	 }
     }
   return 1;
}
