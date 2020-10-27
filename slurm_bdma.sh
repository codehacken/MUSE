#!/bin/bash
# set -vx

# Run all experiments related to BiFNN for word alignment.
SEED=0
INPUT_FILE=${1:-"jobs/test.jobs"}
TRAIN_TYPE=${2:-"supervised"} # Choose between supervised and unsupervised.
ITERATE=${3:-"False"}
DICO_TRAIN=${4:-"combined"} # Change from "default"
LOSS=${5:-"m"} # m - MSE, r - rcsls, mr - MSE+RCSLS.
MEM=40000
WORD_VECTORS_LOC="data/word_vectors"

# Get VOCAB_SIZE.
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
            echo "JOB: $SRC_LANG, $TGT_LANG, $SRC_EMB, $TGT_EMB, $N, $V"
            BASE_NAME="$SRC_LANG-$TGT_LANG-$N-$V-${ITERATE}-${DICO_TRAIN}-${LOSS}-${TRAIN_TYPE}"

            # Baseline alignment.
            if [[ ( "$TRAIN_TYPE" == "supervised" ) ]]
            then
                sbatch --account=pi_ferraro --job-name=${BASE_NAME}-b \
                       --mem=$MEM --qos=medium+ bidnn.slurm $SRC_LANG $TGT_LANG $SRC_EMB $TGT_EMB \
                       $N "baseline" $SEED $ITERATE $V $DICO_TRAIN $LOSS

                sbatch --account=pi_ferraro --job-name=${BASE_NAME}-bdma \
                       --mem=$MEM --qos=medium+ bidnn.slurm $SRC_LANG $TGT_LANG $SRC_EMB $TGT_EMB $N \
                       "bdma" $SEED $ITERATE $V $DICO_TRAIN $LOSS

                sbatch --account=pi_ferraro --job-name=${BASE_NAME}-bdsh \
                       --mem=$MEM --qos=medium+ bidnn.slurm $SRC_LANG $TGT_LANG $SRC_EMB $TGT_EMB $N \
                       "bdma-shared" $SEED $ITERATE $V $DICO_TRAIN $LOSS
            fi

            if [[ ( "$TRAIN_TYPE" == "unsupervised" ) ]]
            then
                sbatch --account=pi_ferraro --job-name=${BASE_NAME}-b \
                       --mem=$MEM --qos=medium+ bidnn_unsup.slurm $SRC_LANG $TGT_LANG $SRC_EMB $TGT_EMB \
                       $N "baseline" $SEED $ITERATE $V $DICO_TRAIN $LOSS

                sbatch --account=pi_ferraro --job-name=${BASE_NAME}-bdma \
                       --mem=$MEM --qos=medium+ bidnn_unsup.slurm $SRC_LANG $TGT_LANG $SRC_EMB $TGT_EMB $N \
                       "bdma" $SEED $ITERATE $V $DICO_TRAIN $LOSS

                sbatch --account=pi_ferraro --job-name=${BASE_NAME}-bdsh \
                       --mem=$MEM --qos=medium+ bidnn_unsup.slurm $SRC_LANG $TGT_LANG $SRC_EMB $TGT_EMB $N \
                       "bdma-shared" $SEED $ITERATE $V $DICO_TRAIN $LOSS
            fi
        done <<< "`tail -n+3 $INPUT_FILE`"
    done
done
