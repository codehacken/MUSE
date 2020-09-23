#!/bin/bash

ORIGINAL_DICT_LOC=data/crosslingual/dictionaries
COMB_DICT_LOC=data/crosslingual/combined_dictionaries
WORD=$1

ORIGINAL_TR_NUM=`grep -sir "$WORD " $ORIGINAL_DICT_LOC/en-es.0-5000.txt  | wc -l`
ORIGINAL_TE_NUM=`grep -sir "$WORD " $ORIGINAL_DICT_LOC/en-es.5000-6500.txt  | wc -l`

COMB_TR_NUM=`grep -sir "$WORD " $COMB_DICT_LOC/en-es.0-5000.txt  | wc -l`
COMB_TE_NUM=`grep -sir "$WORD " $COMB_DICT_LOC/en-es.5000-6500.txt  | wc -l`

echo "Word: $WORD"
echo "Original Dictionary: TR: $ORIGINAL_TR_NUM, TE: $ORIGINAL_TE_NUM"
echo "Combined Dictionary: TR: $COMB_TR_NUM, TE: $COMB_TE_NUM"
