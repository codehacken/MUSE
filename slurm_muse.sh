#!/bin/bash
# set -vx

# Run all experiments related to BiFNN for word alignment.
SEED=0
INPUT_FILE=$1
TRAIN_TYPE=${2:-"supervised"} # Choose between supervised and unsupervised.
ITERATE=${3:-"False"}
DICO_TRAIN=${4:-"combined"} # Change from "default"
MEM=40000

# Get VOCAB_SIZE.
WORD_VECTORS_LOC="data/word_vectors"
VOCAB_SIZE=`head -1 $INPUT_FILE`
VOCAB_SIZE=($VOCAB_SIZE)

# Get LAYER SIZE.
NUM_LAYERS=`head -2 $INPUT_FILE | tail -1`
NUM_LAYERS=($NUM_LAYERS)

for V in "${VOCAB_SIZE[@]}" # Different VOCAB sizes.
do
    for N in "${NUM_LAYERS[@]}" # Different Network sizes.
    do
        # Start processing.
        while IFS= read -r LINE
        do
            # Conditions.
            SRC_LANG=`echo $LINE | awk -F " " '{print $1}'`
            TGT_LANG=`echo $LINE | awk -F " " '{print $2}'`
            SRC_EMB="${WORD_VECTORS_LOC}/${SRC_LANG}/wiki.${SRC_LANG}.bin"
            TGT_EMB="${WORD_VECTORS_LOC}/${TGT_LANG}/wiki.${TGT_LANG}.bin"
            echo "JOB: $SRC_LANG, $TGT_LANG, $SRC_EMB, $TGT_EMB, $N, $V, $TRAIN_TYPE"

            # Baseline alignment.
            # Supervised jobs.
            if [[ ( "$ITERATE" == "False" ) ]]
            then
                sbatch --account=pi_ferraro --job-name=$SRC_LANG-$TGT_LANG-muse2017-$TRAIN_TYPE --mem=$MEM --qos=medium+ \
                       muse.slurm $SRC_LANG $TGT_LANG $SRC_EMB $TGT_EMB $NUM_LAYERS "muse" $SEED $ITERATE \
                       $VOCAB_SIZE $DICO_TRAIN $TRAIN_TYPE
            fi

            # Semi-supervised.
            if [[ ( "$ITERATE" == "True" ) ]]
            then
                sbatch --account=pi_ferraro --job-name=$SRC_LANG-$TGT_LANG-muse-semi-$TRAIN_TYPE --mem=$MEM --qos=medium+ \
                       muse.slurm $SRC_LANG $TGT_LANG $SRC_EMB $TGT_EMB $NUM_LAYERS "muse_semi" $SEED $ITERATE \
                       $VOCAB_SIZE $DICO_TRAIN  $TRAIN_TYPE
            fi
        done <<< "`tail -n+3 $INPUT_FILE`"
    done
done
