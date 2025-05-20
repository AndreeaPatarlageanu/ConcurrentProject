/******************************************************************
  Copyright 2006 Michael Farrar.  
  Freely distributed under the BSD open source license.
  See COPYRIGHT file for details.
*******************************************************************/

/*
  Written by Michael Farrar, 2006.
  For information, contact Sean Eddy (sean@eddylab.org).
*/

#ifndef INCLUDE_SWWOZNIAK_H
#define INCLUDE_SWWOZNIAK_H

#include "swsse2.h"
#include "fastalib.h"

#define MATRIX_ROW_SIZE 32

SW_DATA *
swWozniakInit (unsigned char   *querySeq,
               int              queryLength,
               signed char     *matrix);


void 
swWozniakScan (unsigned char   *querySeq,
               int              queryLength,
               FASTA_LIB       *dbLib,
               void            *swData,
               SEARCH_OPTIONS  *options,
               SCORE_LIST      *scores);

void
swWozniakComplete (SW_DATA *pSwData);


#endif /* INCLUDE_SWWOZNIAK_H */
