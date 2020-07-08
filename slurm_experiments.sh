#!/bin/bash
# set -vx

# Run all experiments related to BiFNN for word alignment.
SEED=0
input=$1
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
    echo "JOB: $SRC_LANG, $TGT_LANG, $SRC_EMB, $TGT_EMB"

    # Baseline alignment.
    sbatch --account=pi_ferraro --job-name=$SRC_LANG-$TGT_LANG-$NUM_LAYERS-b --mem=$MEM --qos=medium+ \
           bidnn.slurm $SRC_LANG $TGT_LANG $SRC_EMB $TGT_EMB $NUM_LAYERS "baseline" $SEED

    sbatch --account=pi_ferraro --job-name=$SRC_LANG-$TGT_LANG-$NUM_LAYERS-bdma --mem=$MEM --qos=medium+ \
           bidnn.slurm $SRC_LANG $TGT_LANG $SRC_EMB $TGT_EMB $NUM_LAYERS "bdma" $SEED

    sbatch --account=pi_ferraro --job-name=$SRC_LANG-$TGT_LANG-$NUM_LAYERS-bdsh --mem=$MEM --qos=medium+ \
           bidnn.slurm $SRC_LANG $TGT_LANG $SRC_EMB $TGT_EMB $NUM_LAYERS "bdma-shared" $SEED

done < "$input"
