#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=output/slurm-%x.out
#SBATCH --error=output/slurm-%x.err

# This for large jobs. #SBATCH --gres=gpu:1
# This for large jobs. #SBATCH --partition=gpu2018

# #SBATCH --gres=gpu:2
# #SBATCH --partition=gpu


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

# Pre-defined.
N_HID_DIM=4096
MAX_VOCAB=200000 # Original is 400000
NUM_EPOCHS=25
MAX_CLIP_WEIGHTS=0

# Verification...
echo "Alignment Languages: Source: $SRC_LANG Target: $TGT_LANG"
echo "Embeddings: Source: $SRC_EMB Target: $TGT_EMB Maximum Vocab Size: $MAX_VOCAB"
echo "Job Type: $JOB_TYPE, Vocab Size: ${VOCAB_SIZE}"
echo "Hyperparameters: Seed: $SEED"

# Once Manifold Alignment is complete.
echo "Model training..."
NAME="${SRC_LANG}_${TGT_LANG}_${VOCAB_SIZE}_${JOB_TYPE}"
EXP_PATH=/scratch/$SLURM_JOB_ID
EMB_LOC=/scratch/$SLURM_JOB_ID/debug/${NAME}

if [[ ( "$JOB_TYPE" == "muse" ) ]]
then
    echo "Training baseline MUSE-SUPERVISED model..."

    # n-refinement is 0 by default for basic supervised model.
    python supervised.py --src_lang $SRC_LANG --tgt_lang $TGT_LANG \
                         --src_emb $SRC_EMB --tgt_emb $TGT_EMB --n_refinement 0 \
                         --dico_train $DICO_TRAIN --seed $SEED \
                         --export pth --exp_id ${NAME} --exp_path ${EXP_PATH} \
                         --dico_eval combined  2> data/muse_baseline_results/${NAME}.results
fi

if [[ ( "$JOB_TYPE" == "muse_semi" ) ]]
then
    echo "Training MUSE Semisupervised model..."
    python supervised.py --src_lang $SRC_LANG --tgt_lang $TGT_LANG \
                         --src_emb $SRC_EMB --tgt_emb $TGT_EMB --n_refinement 5 \
                         --dico_train $DICO_TRAIN --seed $SEED \
                         --export pth --exp_id ${NAME} --exp_path ${EXP_PATH} \
                         --dico_eval combined  2> data/muse_baseline_results/${NAME}.results
fi

echo "Training Completed... Starting Evaluation..."

# Evaluate.
python evaluate.py --src_lang $SRC_LANG --tgt_lang $TGT_LANG \
                   --src_emb ${EMB_LOC}/vectors-${SRC_LANG}-f.pth \
                   --tgt_emb ${EMB_LOC}/vectors-${TGT_LANG}-f.pth \
                   --max_vocab $MAX_VOCAB --exp_id eval_${NAME}_f --cuda True \
                   --exp_path ${EXP_PATH} --dico_eval combined 2> data/muse_baseline_results/eval_${NAME}_f.results
