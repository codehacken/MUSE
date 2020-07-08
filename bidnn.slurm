#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --output=output/slurm-%x.out
#SBATCH --error=output/slurm-%x.err

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

# Verification...
echo "Alignment Languages: Source: $SRC_LANG Target: $TGT_LANG"
echo "Embeddings: Source: $SRC_EMB Target: $TGT_EMB"
echo "Network Size: $NUM_LAYERS"
echo "Job Type: $JOB_TYPE"
echo "Hyperparameters: Seed: $SEED"

# Once Manifold Alignment is complete.
echo "Model training..."
NAME="${SRC_LANG}_${TGT_LANG}_${NUM_LAYERS}_${JOB_TYPE}"
if [[ ( "$JOB_TYPE" == "baseline" ) ]]
then
    echo "Training baseline model..."
    # python bdma_sup.py --src_lang $SRC_LANG --tgt_lang $TGT_LANG \
    #                    --src_emb $SRC_EMB --tgt_emb $TGT_EMB --n_refinement 20 \
    #                    --dico_train default --n_layers $NUM_LAYERS --n_hid_dim 4096 \
    #                    --batch_size 256 --seed $SEED --export pth \
    #                    --exp_id ${NAME} 2> data/results/${NAME}.results
fi

if [[ ( "$JOB_TYPE" == "bdma" ) ]]
then
    echo "Training BDMA model..."
    # python bdma_sup.py --src_lang $SRC_LANG --tgt_lang $TGT_LANG \
    #                    --src_emb $SRC_EMB --tgt_emb $TGT_EMB --n_refinement 20 \
    #                    --dico_train default --n_layers $NUM_LAYERS --n_hid_dim 4096 \
    #                    --batch_size 256 --seed $SEED --export pth --bidirectional True --shared False \
    #                    --exp_id ${NAME} 2> data/results/${NAME}.results
fi

if [[ ( "$JOB_TYPE" == "bdma-shared" ) ]]
then
    echo "Training BDMA shared model..."
    # python bdma_sup.py --src_lang $SRC_LANG --tgt_lang $TGT_LANG \
    #                    --src_emb $SRC_EMB --tgt_emb $TGT_EMB --n_refinement 20 \
    #                    --dico_train default --n_layers $NUM_LAYERS --n_hid_dim 4096 \
    #                    --batch_size 256 --seed $SEED --export pth --bidirectional True --shared True \
    #                    --exp_id ${NAME} 2> data/results/${NAME}.results
fi
echo "Training Completed... Starting Evaluation..."

# Evaluate.
# python evaluate.py --src_lang $SRC_LANG --tgt_lang $TGT_LANG \
#                    --src_emb dumped/debug/${NAME}/vectors-${SRC_LANG}-f.pth \
#                    --tgt_emb dumped/debug/${NAME}/vectors-${TGT_LANG}-f.pth \
#                    --max_vocab 400000 --exp_id eval_${NAME}_f \
#                    2> data/results/eval_${NAME}_f.results
#
# python evaluate.py --src_lang $TGT_LANG --tgt_lang $SRC_LANG \
#                    --tgt_emb dumped/debug/${NAME}/vectors-${SRC_LANG}-b.pth \
#                    --src_emb dumped/debug/${NAME}/vectors-${TGT_LANG}-b.pth \
#                    --max_vocab 400000 --exp_id eval_${NAME}_b \
#                    2> data/results/eval_${NAME}_b.results
