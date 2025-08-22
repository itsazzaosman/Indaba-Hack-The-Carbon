#!/bin/bash

# Launch script for 8-GPU distributed training
# This script sets up the environment and launches distributed training

# Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# Training parameters
TRAIN_CSV="train.csv"           # Update with your actual path
VAL_CSV="val.csv"               # Update with your actual path
ROOT_DIR="."                     # Update with your actual path
BATCH_SIZE=8                     # Batch size per GPU
EPOCHS=50                        # Number of training epochs
LEARNING_RATE=1e-4              # Learning rate
WEIGHT_DECAY=1e-4               # Weight decay
FEATURE_DIM=1024                # Feature dimension
NUM_WORKERS=4                   # Number of workers per GPU
SAVE_DIR="./checkpoints"        # Checkpoint save directory

# Check if required files exist
if [ ! -f "$TRAIN_CSV" ]; then
    echo "Error: Training CSV not found: $TRAIN_CSV"
    echo "Please update the TRAIN_CSV variable in this script"
    exit 1
fi

if [ ! -f "$VAL_CSV" ]; then
    echo "Error: Validation CSV not found: $VAL_CSV"
    echo "Please update the VAL_CSV variable in this script"
    exit 1
fi

if [ ! -d "$ROOT_DIR" ]; then
    echo "Error: Root directory not found: $ROOT_DIR"
    echo "Please update the ROOT_DIR variable in this script"
    exit 1
fi

# Create save directory
mkdir -p "$SAVE_DIR"

echo "=== 8-GPU XGBoost Biomass Training Launch ==="
echo "Training CSV: $TRAIN_CSV"
echo "Validation CSV: $VAL_CSV"
echo "Root Directory: $ROOT_DIR"
echo "Batch Size per GPU: $BATCH_SIZE"
echo "Total Batch Size: $((BATCH_SIZE * 8))"
echo "Epochs: $EPOCHS"
echo "Learning Rate: $LEARNING_RATE"
echo "Feature Dimension: $FEATURE_DIM"
echo "Save Directory: $SAVE_DIR"
echo ""

# Check GPU availability
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits

echo ""
echo "Launching distributed training with 8 GPUs..."
echo ""

# Launch distributed training using torchrun
# Get the absolute path to the current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=12355 \
    "$SCRIPT_DIR/train_8gpu.py" \
    --train-csv "$TRAIN_CSV" \
    --val-csv "$VAL_CSV" \
    --root-dir "$ROOT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LEARNING_RATE" \
    --weight-decay "$WEIGHT_DECAY" \
    --feature-dim "$FEATURE_DIM" \
    --num-workers "$NUM_WORKERS" \
    --save-dir "$SAVE_DIR"

echo ""
echo "Training completed!"
echo "Checkpoints saved to: $SAVE_DIR"
