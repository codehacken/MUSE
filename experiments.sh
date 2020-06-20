#!/bin/bash
set -vx

# Run all experiments related to BiFNN for word alignment.
# rm data/results/all_jobs.results
# FINAL=all_jobs_fr_en_1.results
FINAL=${2:-"all_jobs_fr_en_1.results"}
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
    NAME_B="${SRC_LANG}_${TGT_LANG}_${NUM_LAYERS}_baseline"
    python bdma_sup.py --src_lang $SRC_LANG --tgt_lang $TGT_LANG \
               --src_emb $SRC_EMB --tgt_emb $TGT_EMB --n_refinement 20 \
               --dico_train default --n_layers $NUM_LAYERS --n_hid_dim 4096 \
               --batch_size 256 --seed $SEED --export pth \
               --exp_id ${NAME_B} 2> data/results/${NAME_B}.results

    # BDMA.
    NAME_BDMA="${SRC_LANG}_${TGT_LANG}_${NUM_LAYERS}_bdma"
    python bdma_sup.py --src_lang $SRC_LANG --tgt_lang $TGT_LANG \
               --src_emb $SRC_EMB --tgt_emb $TGT_EMB --n_refinement 20 \
               --dico_train default --n_layers $NUM_LAYERS --n_hid_dim 4096 \
               --batch_size 256 --seed $SEED --export pth --bidirectional True --shared False \
               --exp_id ${NAME_BDMA} 2> data/results/${NAME_BDMA}.results

    # BDMA Shared.
    NAME_BDMA_S="${SRC_LANG}_${TGT_LANG}_${NUM_LAYERS}_bdma_shared"
    python bdma_sup.py --src_lang $SRC_LANG --tgt_lang $TGT_LANG \
               --src_emb $SRC_EMB --tgt_emb $TGT_EMB --n_refinement 20 \
               --dico_train default --n_layers $NUM_LAYERS --n_hid_dim 4096 \
               --batch_size 256 --seed $SEED --export pth --bidirectional True --shared True \
               --exp_id ${NAME_BDMA_S} 2> data/results/${NAME_BDMA_S}.results

    echo "Waiting..."
    wait

    # Evaluate.
    python evaluate.py --src_lang $SRC_LANG --tgt_lang $TGT_LANG \
                       --src_emb dumped/debug/${NAME_B}/vectors-${SRC_LANG}_f.pth \
                       --tgt_emb dumped/debug/${NAME_B}/vectors-${TGT_LANG}_f.pth \
                       --max_vocab 400000 --exp_id eval_${NAME_B}_f \
                       2> data/results/eval_${NAME_B}_f.results

    echo -e "\n$NAME_B:" >>  data/results/$FINAL
    tail -35 data/results/eval_${NAME_B}_f.results >> data/results/$FINAL

    python evaluate.py --src_lang $TGT_LANG --tgt_lang $SRC_LANG \
                       --tgt_emb dumped/debug/${NAME_B}/vectors-${SRC_LANG}_b.pth \
                       --src_emb dumped/debug/${NAME_B}/vectors-${TGT_LANG}_b.pth \
                       --max_vocab 400000 --exp_id eval_${NAME_B}_b \
                       2> data/results/eval_${NAME_B}_b.results

    echo -e "\n$NAME_B:" >>  data/results/$FINAL
    tail -35 data/results/eval_${NAME_B}_b.results >> data/results/$FINAL

    # BDMA
    python evaluate.py --src_lang $SRC_LANG --tgt_lang $TGT_LANG \
                       --src_emb dumped/debug/${NAME_BDMA}/vectors-${SRC_LANG}_f.pth \
                       --tgt_emb dumped/debug/${NAME_BDMA}/vectors-${TGT_LANG}_f.pth \
                       --max_vocab 400000 --exp_id eval_${NAME_BDMA}_f \
                       2> data/results/eval_${NAME_BDMA}_f.results

    echo -e "\n$NAME_BDMA:" >>  data/results/$FINAL
    tail -35 data/results/eval_${NAME_BDMA}_f.results >> data/results/$FINAL

    python evaluate.py --tgt_lang $SRC_LANG --src_lang $TGT_LANG \
                       --tgt_emb dumped/debug/${NAME_BDMA}/vectors-${SRC_LANG}_b.pth \
                       --src_emb dumped/debug/${NAME_BDMA}/vectors-${TGT_LANG}_b.pth \
                       --max_vocab 400000 --exp_id eval_${NAME_BDMA}_b \
                       2> data/results/eval_${NAME_BDMA}_b.results

    echo -e "\n$NAME_BDMA:" >>  data/results/$FINAL
    tail -35 data/results/eval_${NAME_BDMA}_b.results >> data/results/$FINAL

    # BDMA Shared
    python evaluate.py --src_lang $SRC_LANG --tgt_lang $TGT_LANG \
                       --src_emb dumped/debug/${NAME_BDMA_S}/vectors-${SRC_LANG}_f.pth \
                       --tgt_emb dumped/debug/${NAME_BDMA_S}/vectors-${TGT_LANG}_f.pth \
                       --max_vocab 400000 --exp_id eval_${NAME_BDMA_S}_f \
                       2> data/results/eval_${NAME_BDMA_S}_f.results

    echo -e "\n$NAME_BDMA_S:" >>  data/results/$FINAL
    tail -35 data/results/eval_${NAME_BDMA_S}_f.results >> data/results/$FINAL

    python evaluate.py --tgt_lang $SRC_LANG --src_lang $TGT_LANG \
                       --tgt_emb dumped/debug/${NAME_BDMA_S}/vectors-${SRC_LANG}_b.pth \
                       --src_emb dumped/debug/${NAME_BDMA_S}/vectors-${TGT_LANG}_b.pth \
                       --max_vocab 400000 --exp_id eval_${NAME_BDMA_S}_b \
                       2> data/results/eval_${NAME_BDMA_S}_b.results

    echo -e "\n$NAME_BDMA_S:" >>  data/results/$FINAL
    tail -35 data/results/eval_${NAME_BDMA_S}_b.results >> data/results/$FINAL

done < "$input"
