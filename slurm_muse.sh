#!/bin/bash
# set -vx

# Run all experiments related to BiFNN for word alignment.
SEED=0
input=$1
ITERATE=${2:-"False"}
DICO_TRAIN=${3:-"combined"} # Change from "default"
MEM=40000

# Start processing.
while IFS= read -r LINE
do
    # Conditions.
    SRC_LANG=`echo $LINE | awk -F " " '{print $1}'`
    TGT_LANG=`echo $LINE | awk -F " " '{print $2}'`
    SRC_EMB=`echo $LINE | awk -F " " '{print $3}'`
    TGT_EMB=`echo $LINE | awk -F " " '{print $4}'`
    NUM_LAYERS=`echo $LINE | awk -F " " '{print $5}'`
    VOCAB_SIZE=`echo $LINE | awk -F " " '{print $6}'`
    echo "JOB: $SRC_LANG, $TGT_LANG, $SRC_EMB, $TGT_EMB, $NUM_LAYERS, $VOCAB_SIZE"

    # Baseline alignment.
    # Supervised jobs.
    if [[ ( "$ITERATE" == "False" ) ]]
    then
        sbatch --account=pi_ferraro --job-name=$SRC_LANG-$TGT_LANG-muse2017 --mem=$MEM --qos=medium+ \
               muse.slurm $SRC_LANG $TGT_LANG $SRC_EMB $TGT_EMB $NUM_LAYERS "muse" $SEED $ITERATE $VOCAB_SIZE $DICO_TRAIN
    fi

    # Semi-supervised.
    if [[ ( "$ITERATE" == "True" ) ]]
    then
        sbatch --account=pi_ferraro --job-name=$SRC_LANG-$TGT_LANG-muse-semi --mem=$MEM --qos=medium+ \
               muse.slurm $SRC_LANG $TGT_LANG $SRC_EMB $TGT_EMB $NUM_LAYERS "muse_semi" $SEED $ITERATE $VOCAB_SIZE $DICO_TRAIN
    fi
done < "$input"
