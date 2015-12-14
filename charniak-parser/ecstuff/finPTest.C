#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <signal.h>
#include <assert.h>
#include "ECArgs.h"
#include "utils.h"
#include "ECString.h"
#include "ParseStats.h"


#define MAXN 10


// Misc utility functions.
//

void
usage( char * program )
{
    cerr << program << " usage: " << program
      << " [-w Wordtype] [ <file_prefix> ]" << endl;
    exit( 1 );
}


int
main(int argc, char *argv[])
{
   ECArgs args(argc, argv);
   if( args.nargs() > 1 )	// max one filename allowed.
     usage( *argv );
   
   ECString path = args.arg(0);

   cout.precision(5);

   int totSents = 0;
   int totUnparsed = 0;
   
   ECString fileString(path);
   fileString += "PStats.txt";
   ECString      parseStatsString( fileString );
   ifstream    parseStatsStream( parseStatsString.c_str() );
   if( !parseStatsStream ) error( "unable to open parseStats stream." );

   ParseStats totals[500];
   while( parseStatsStream )
     {
       int len;
       parseStatsStream >> len;
       if(!parseStatsStream) break;
       ParseStats tot;
       parseStatsStream >> tot;
       totSents++;
       totals[len] += tot;
     }
   cout << "Tot sentences: " << totSents << "\n";

   for(int i = 0 ; i < 500 ; i++)
     {
       if(totals[i].numInGold == 0) continue;
       else cout << i << "\t" << totals[i].precision() << "\t"
		 << totals[i].recall() << "\t" << totals[i].fMeasure() << endl;
     }
   return 0;  
 }

