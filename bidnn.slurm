#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --output=output/slurm-%x.out
#SBATCH --error=output/slurm-%x.err

# Large Jobs
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu2018

# Small jobs.
# SBATCH --gres=gpu:2
# SBATCH --partition=gpu
# SBATCH --exclude=gpunode[111-116]

# Set the environment.
source deactivate # Remove previous environments.
source activate cuda9-py38-pytorch1.5

# Execute the code.
set -o xtrace
SRC_LANG=$1
TGT_LANG=$2
SRC_EMB=$3
TGT_EMB=$4
NUM_LAYERS=$5
JOB_TYPE=$6
SEED=$7
ITERATE=$8
VOCAB_SIZE=$9
DICO_TRAIN=${10:-"default"}
LOSS=${11:-"m"}

# Pre-defined.
N_HID_DIM=2048
MAX_VOCAB=200000 # Original is 400000
NUM_EPOCHS=30
MAX_CLIP_WEIGHTS=0

# Verification...
echo "Alignment Languages: Source: $SRC_LANG Target: $TGT_LANG"
echo "Embeddings: Source: $SRC_EMB Target: $TGT_EMB Maximum Vocab Size: $MAX_VOCAB"
echo "Network Size: $NUM_LAYERS, Hidden Layer: $N_HID_DIM"
echo "Job Type: $JOB_TYPE and Loss: $LOSS"
echo "Hyperparameters: Seed: $SEED"

# Once Manifold Alignment is complete.
echo "Model training..."
NAME="${SRC_LANG}_${TGT_LANG}_${NUM_LAYERS}_${VOCAB_SIZE}_${ITERATE}_${DICO_TRAIN}_${LOSS}_${JOB_TYPE}"
EXP_PATH=/scratch/$SLURM_JOB_ID
EMB_LOC=/scratch/$SLURM_JOB_ID/debug/${NAME}

if [[ ( "$JOB_TYPE" == "baseline" ) ]]
then
    echo "Training baseline model..."
    python bdma_sup.py --src_lang $SRC_LANG --tgt_lang $TGT_LANG \
                       --src_emb $SRC_EMB --tgt_emb $TGT_EMB --n_refinement $NUM_EPOCHS --vocab_size $VOCAB_SIZE\
                       --dico_train $DICO_TRAIN --n_layers $NUM_LAYERS --n_hid_dim $N_HID_DIM --map_clip_weights $MAX_CLIP_WEIGHTS \
                       --batch_size 128 --seed $SEED --export pth --shared True --iterative $ITERATE --loss $LOSS \
                       --exp_id ${NAME} --exp_path ${EXP_PATH} --dico_eval combined  2> data/results/${NAME}.results
fi

if [[ ( "$JOB_TYPE" == "bdma" ) ]]
then
    echo "Training BDMA model..."
    python bdma_sup.py --src_lang $SRC_LANG --tgt_lang $TGT_LANG --map_clip_weights $MAX_CLIP_WEIGHTS \
                       --src_emb $SRC_EMB --tgt_emb $TGT_EMB --n_refinement $NUM_EPOCHS --vocab_size $VOCAB_SIZE \
                       --dico_train $DICO_TRAIN --n_layers $NUM_LAYERS --n_hid_dim $N_HID_DIM --iterative $ITERATE \
                       --batch_size 128 --seed $SEED --export pth --bidirectional True --shared False --loss $LOSS  \
                       --exp_id ${NAME} --exp_path ${EXP_PATH} --dico_eval combined  2> data/results/${NAME}.results
fi

if [[ ( "$JOB_TYPE" == "bdma-shared" ) ]]
then
    echo "Training BDMA shared model..."
    python bdma_sup.py --src_lang $SRC_LANG --tgt_lang $TGT_LANG --vocab_size $VOCAB_SIZE  --map_clip_weights $MAX_CLIP_WEIGHTS \
                       --src_emb $SRC_EMB --tgt_emb $TGT_EMB --n_refinement $NUM_EPOCHS \
                       --dico_train $DICO_TRAIN --n_layers $NUM_LAYERS --n_hid_dim $N_HID_DIM --iterative $ITERATE \
                       --batch_size 128 --seed $SEED --export pth --bidirectional True --shared True --loss $LOSS  \
                       --exp_id ${NAME} --exp_path ${EXP_PATH} --dico_eval combined  2> data/results/${NAME}.results
fi
echo "Training Completed... Starting Evaluation..."

# Evaluate.
python evaluate.py --src_lang $SRC_LANG --tgt_lang $TGT_LANG \
                   --src_emb ${EMB_LOC}/vectors-${SRC_LANG}-f.pth \
                   --tgt_emb ${EMB_LOC}/vectors-${TGT_LANG}-f.pth \
                   --max_vocab $MAX_VOCAB --exp_id eval_${NAME}_f --cuda True  --loss $LOSS \
                   --exp_path ${EXP_PATH} --dico_eval combined 2> data/results/eval_${NAME}_f.results

python evaluate.py --src_lang $TGT_LANG --tgt_lang $SRC_LANG \
                   --tgt_emb ${EMB_LOC}/vectors-${SRC_LANG}-b.pth \
                   --src_emb ${EMB_LOC}/vectors-${TGT_LANG}-b.pth \
                   --max_vocab $MAX_VOCAB --exp_id eval_${NAME}_b --cuda True  --loss $LOSS \
                   --exp_path ${EXP_PATH} --dico_eval combined 2> data/results/eval_${NAME}_b.results
