#!/bin/bash
set -vx

# Run all experiments related to BiFNN for word alignment.
SEED=0
input=$1
while IFS= read -r LINE
do
    # Conditions.
    SRC_LANG=`echo $LINE | awk -F " " '{print $1}'`
    TGT_LANG=`echo $LINE | awk -F " " '{print $2}'`
    SRC_EMB=`echo $LINE | awk -F " " '{print $3}'`
    TGT_EMB=`echo $LINE | awk -F " " '{print $4}'`
    NUM_LAYERS=`echo $LINE | awk -F " " '{print $5}'`
    echo "JOB: $SRC_LANG, $TGT_LANG, $SRC_EMB, $TGT_EMB"

    # Baseline.
    NAME="${SRC_LANG}_${TGT_LANG}_${NUM_LAYERS}"
    python bdma_sup.py --src_lang $SRC_LANG --tgt_lang $TGT_LANG \
               --src_emb $SRC_EMB --tgt_emb $TGT_EMB --n_refinement 5 \
               --dico_train default --n_layers $NUM_LAYERS --n_hid_dim 4096 \
               --batch_size 256 --seed $SEED --exp_id $NAME 2> data/results/${NAME}.results

    # Evaluate.
    python evaluate.py --src_lang $SRC_LANG --tgt_lang $TGT_LANG \
                       --src_emb $SRC_LANG --tgt_emb $TGT_LANG \
                       --max_vocab 200000 2> data/results/eval_${NAME}.results

done < "$input"
