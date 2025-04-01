#!/bin/bash

# Kill any previous python processes (if needed)
echo "Stopping any existing training processes..."
pkill -9 python || true
sleep 2

# Set up environment for distributed training
# Start with 4 GPUs instead of 8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export WORLD_SIZE=6
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Print environment variables for debugging
echo "Starting training with:"
echo "WORLD_SIZE=$WORLD_SIZE"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"

# Define GPT-2 model size
# GPT2_MODEL="gpt2"  # Options: "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"

# Run processes
for ((i=0; i<$WORLD_SIZE; i++)); do
    export RANK=$i
    export LOCAL_RANK=$i
    
    echo "Starting process RANK=$RANK on GPU LOCAL_RANK=$LOCAL_RANK"
    python -m segm.train --log-dir seg_tiny_mask_pascal --dataset pascal_context \
      --backbone vit_tiny_patch16_384 --decoder mask_transformer \
      # --gpt2-model-name $GPT2_MODEL &

    # Add a longer delay to prevent race conditions
    sleep 3
    echo "Process $i started."
done

# echo "All processes launched with GPT-2 model: $GPT2_MODEL. Waiting for training to complete..."
echo "All processes launched. Waiting for training to complete..."
# Wait for all processes to complete
wait