#!/bin/bash
# set -vx
# Extract the results from the eval file generated at the end of the run.

RESULTS_LOC=$1
EXP_NAME=$2
NUM_LAYERS=${3:-"0"}
VOCAB_SIZE=${4:-"0"}

# Set for different combinations.
LANG_1=( "en" )
LANG_2=( "es" "it" "de" "fr" "hi" "ru" "ja" "pt")

echo "Experiment: $EXP_NAME"
for s in ${LANG_1[@]}; do
    for t in ${LANG_2[@]}; do
        # S->T
        S_T_F="${RESULTS_LOC}/eval_${s}_${t}_${NUM_LAYERS}_${VOCAB_SIZE}_bdma-shared_f.results"
        if [ -f "$S_T_F" ]; then
            PREC_F=`grep -sir "Precision at k = 1:" $S_T_F | tail -1 | awk -F ": " '{print $2}'`
        fi

        S_T_B="${RESULTS_LOC}/eval_${s}_${t}_${NUM_LAYERS}_${VOCAB_SIZE}_bdma-shared_b.results"
        if [ -f "$S_T_B" ]; then
            PREC_B=`grep -sir "Precision at k = 1:" $S_T_B | tail -1 | awk -F ": " '{print $2}'`
        fi
        echo "$s -> $t: Forward: $PREC_F Reverse: $PREC_B"

        # T->S
        T_S_F="${RESULTS_LOC}/eval_${t}_${s}_${NUM_LAYERS}_${VOCAB_SIZE}_bdma-shared_f.results"
        if [ -f "$T_S_F" ]; then
            PREC_F=`grep -sir "Precision at k = 1:" $T_S_F | tail -1 | awk -F ": " '{print $2}'`
        fi

        T_S_B="${RESULTS_LOC}/eval_${t}_${s}_${NUM_LAYERS}_${VOCAB_SIZE}_bdma-shared_b.results"
        if [ -f "$T_S_B" ]; then
            PREC_B=`grep -sir "Precision at k = 1:" $T_S_B | tail -1 | awk -F ": " '{print $2}'`
        fi
        echo "$t -> $s: Forward: $PREC_F Reverse: $PREC_B"
    done
done
echo ""
