/******************************************************************
  Copyright 2006 Michael Farrar.  
  Freely distributed under the BSD open source license.
  See COPYRIGHT file for details.
*******************************************************************/

/*
  Written by Michael Farrar, 2006.
  For information, contact Sean Eddy (sean@eddylab.org).
*/

#ifndef INCLUDE_SWSCALAR_H
#define INCLUDE_SWSCALAR_H

#include "swsse2.h"
#include "fastalib.h"

SW_DATA *
swScalarInit (unsigned char   *querySeq,
              int              queryLength,
              signed char     *matrix);


void 
swScalarScan (unsigned char   *querySeq,
              int              queryLength,
              FASTA_LIB       *dbLib,
              void            *swData,
              SEARCH_OPTIONS  *options,
              SCORE_LIST      *scores);


void
swScalarComplete (SW_DATA *pSwData);


#endif /* INCLUDE_SWSCALAR_H */
