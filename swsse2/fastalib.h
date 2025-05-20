/******************************************************************
  Copyright 2006 Michael Farrar.  
  Freely distributed under the BSD open source license.
  See COPYRIGHT file for details.
*******************************************************************/

/*
  Written by Michael Farrar, 2006.
  For information, contact Sean Eddy (sean@eddylab.org).
*/

#ifndef INCLUDE_FASTALIB_H
#define INCLUDE_FASTALIB_H

#include <stdio.h>

#define MAX_SEQ_LENGTH (64 * 1024)

typedef struct {
    char *readBuffer;

    char *seqName;
    unsigned char *seqBuffer;

    int pos;
    int size;

    FILE *fp;

    int sequences;
    int residues;

    int pad;
} FASTA_LIB;

FASTA_LIB *openLib (char *file, int pad);
void closeLib (FASTA_LIB *lib);

unsigned char *nextSeq (FASTA_LIB *lib, int *length);

#define seqName(LIB) (LIB->seqName)

#endif /* INCLUDE_FASTALIB_H */
