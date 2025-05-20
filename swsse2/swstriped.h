/******************************************************************
  Copyright 2006 Michael Farrar.  
  Freely distributed under the BSD open source license.
  See COPYRIGHT file for details.
*******************************************************************/

/*
  Written by Michael Farrar, 2006.
  For information, contact Sean Eddy (sean@eddylab.org).
*/


#ifndef INCLUDE_SWSTRIPED_H
#define INCLUDE_SWSTRIPED_H

#include <emmintrin.h>

#include "swsse2.h"
#include "fastalib.h"

SW_DATA *
swStripedInit (unsigned char   *querySeq,
               int              queryLength,
               signed char     *matrix);


void 
swStripedScan (unsigned char   *querySeq,
               int              queryLength,
               FASTA_LIB       *dbLib,
               void            *swData,
               SEARCH_OPTIONS  *options,
               SCORE_LIST      *scores);

                    
void
swStripedComplete (SW_DATA *pSwData);


#endif /* INCLUDE_SWSTRIPED_H */
