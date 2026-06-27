#!/usr/bin/env zsh
#
# train_pipeline.sh
#
# This script automates the entire fine-tuning pipeline for buddy-cortex on Apple Silicon.
# It handles prerequisite verification, data verification (and automatic conversion if needed),
# lora training execution, and weight fusing.
#

# Exit immediately if a command exits with a non-zero status
set -e

# Unset external VIRTUAL_ENV to avoid uv matching issues if shell has a sibling activated venv
unset VIRTUAL_ENV

# Default configurations
MODEL="mlx-community/gemma-2-9b-it-4bit"
DATA_DIR="./data"
NUM_LAYERS=16
BATCH_SIZE=1             # Keep batch size small for VRAM limits
GRAD_ACCUMULATION_STEPS=4 # Accumulate gradients to simulate batch size of (BATCH_SIZE * GRAD_ACCUMULATION_STEPS)
ITERS=1000
ADAPTER_PATH="adapters"
DRY_RUN=false
MAX_SEQ_LENGTH=2048      # Set to 2048 to cover 99.4% of data while maintaining low memory
VAL_BATCHES=10           # Limit evaluation batches (prevents long evaluation & memory build-up)
GRAD_CHECKPOINT=true     # Enabled by default to reduce memory usage during training

# Helper function to print usage instructions
print_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -m, --model <str>          Base model to use (default: $MODEL)"
    echo "  -d, --data <path>          Directory containing train.jsonl & valid.jsonl (default: $DATA_DIR)"
    echo "  -l, --num-layers <int>     Number of layers to apply LoRA (default: $NUM_LAYERS, use -1 for all)"
    echo "  -b, --batch-size <int>     Batch size for training (default: $BATCH_SIZE)"
    echo "  --grad-accumulation-steps <int>  Number of steps to accumulate gradients (default: $GRAD_ACCUMULATION_STEPS)"
    echo "  -i, --iters <int>          Number of training iterations (default: $ITERS)"
    echo "  -a, --adapter-path <path>  Directory to save/load adapters (default: $ADAPTER_PATH)"
    echo "  --max-seq-length <int>     Maximum sequence length (default: $MAX_SEQ_LENGTH)"
    echo "  --val-batches <int>        Number of validation batches (default: $VAL_BATCHES, use -1 for all)"
    echo "  --no-grad-checkpoint       Disable gradient checkpointing to speed up training if memory is sufficient"
    echo "  --dry-run                  Prepare everything but skip the actual MLX training & fusing steps"
    echo "  -h, --help                 Show this help message"
    echo ""
}

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -m|--model) MODEL="$2"; shift ;;
        -d|--data) DATA_DIR="$2"; shift ;;
        -l|--num-layers) NUM_LAYERS="$2"; shift ;;
        -b|--batch-size) BATCH_SIZE="$2"; shift ;;
        --grad-accumulation-steps) GRAD_ACCUMULATION_STEPS="$2"; shift ;;
        -i|--iters) ITERS="$2"; shift ;;
        -a|--adapter-path) ADAPTER_PATH="$2"; shift ;;
        --max-seq-length) MAX_SEQ_LENGTH="$2"; shift ;;
        --val-batches) VAL_BATCHES="$2"; shift ;;
        --no-grad-checkpoint) GRAD_CHECKPOINT=false ;;
        --dry-run) DRY_RUN=true ;;
        -h|--help) print_usage; exit 0 ;;
        *) echo "[-] Unknown parameter passed: $1"; print_usage; exit 1 ;;
    esac
    shift
done

echo "=========================================================="
echo "      🚀 buddy-cortex MLX Fine-Tuning Pipeline 🚀"
echo "=========================================================="
echo "[*] Base Model:             $MODEL"
echo "[*] Data Directory:         $DATA_DIR"
echo "[*] LoRA Layers:            $NUM_LAYERS"
echo "[*] Batch Size:             $BATCH_SIZE"
echo "[*] Grad Accumulation:      $GRAD_ACCUMULATION_STEPS (Effective Batch Size: $((BATCH_SIZE * GRAD_ACCUMULATION_STEPS)))"
echo "[*] Iterations:             $ITERS"
echo "[*] Adapter Path:           $ADAPTER_PATH"
echo "[*] Max Seq Length:         $MAX_SEQ_LENGTH"
echo "[*] Val Batches:            $VAL_BATCHES"
echo "[*] Grad Checkpoint:        $GRAD_CHECKPOINT"
if [ "$DRY_RUN" = true ]; then
    echo "[*] Mode:                   DRY RUN (Preparation only)"
fi
echo "=========================================================="

# 1. Check if Apple Silicon Mac is being used
if [[ "$(uname -m)" != "arm64" ]]; then
    echo "[!] Warning: This script is optimized for Apple Silicon (arm64) architectures."
    echo "    Performance on non-Apple Silicon platforms may be extremely slow or unsupported."
fi

# 2. Check if uv is installed and dependencies are synced
if ! command -v uv &>/dev/null; then
    echo "[-] Error: 'uv' command not found."
    echo "    Please install uv (https://astral.sh/uv) and try again."
    exit 1
fi

echo "[*] Synchronizing project dependencies with uv..."
uv sync

if ! uv run python -c "import mlx_lm" &>/dev/null; then
    echo "[-] Error: 'mlx-lm' library not found in the uv environment."
    echo "    Please add it using: uv add mlx-lm"
    exit 1
fi
echo "[+] Prerequisite check passed: 'mlx-lm' library is available via uv."

# 3. Verify and prepare dataset
TRAIN_FILE="$DATA_DIR/train.jsonl"
VALID_FILE="$DATA_DIR/valid.jsonl"

if [ ! -f "$TRAIN_FILE" ] || [ ! -f "$VALID_FILE" ]; then
    echo "[!] Warning: Training or validation data not found in '$DATA_DIR'."
    if [ -f "dataset.csv" ]; then
        echo "[*] Found 'dataset.csv'. Automatically running 'convert_dataset.py' to generate data..."
        uv run python convert_dataset.py --output-dir "$DATA_DIR"
    else
        echo "[-] Error: No dataset.csv or processed JSONL files found."
        echo "    Please place 'dataset.csv' in this folder and try again."
        exit 1
    fi
else
    echo "[+] Found processed datasets in '$DATA_DIR'."
fi

# 4. Trigger training
if [ "$DRY_RUN" = true ]; then
    echo "[*] Dry-run enabled. Skipping the actual MLX training step."
    echo "    To run training, execute: "
    echo "    uv run python -m mlx_lm lora --model $MODEL --data $DATA_DIR --train --iters $ITERS --batch-size $BATCH_SIZE --grad-accumulation-steps $GRAD_ACCUMULATION_STEPS --num-layers $NUM_LAYERS --adapter-path $ADAPTER_PATH --max-seq-length $MAX_SEQ_LENGTH --val-batches $VAL_BATCHES$( [ "$GRAD_CHECKPOINT" = true ] && echo " --grad-checkpoint" )"
else
    echo "[*] Phase 1: Initiating MLX LoRA fine-tuning..."
    
    LORA_ARGS=(
        --model "$MODEL"
        --data "$DATA_DIR"
        --train
        --iters "$ITERS"
        --batch-size "$BATCH_SIZE"
        --grad-accumulation-steps "$GRAD_ACCUMULATION_STEPS"
        --num-layers "$NUM_LAYERS"
        --adapter-path "$ADAPTER_PATH"
        --max-seq-length "$MAX_SEQ_LENGTH"
        --val-batches "$VAL_BATCHES"
    )
    if [ "$GRAD_CHECKPOINT" = true ]; then
        LORA_ARGS+=(--grad-checkpoint)
    fi

    uv run python -m mlx_lm lora "${LORA_ARGS[@]}"
    echo "[+] Fine-tuning completed successfully. Adapters saved in: $ADAPTER_PATH"
fi

# 5. Model Merge & Fusing (Removed)
# Native MLX fusing is removed since GGUF conversion is handled separately by convert_gguf.sh.
echo "[*] Training complete. To convert your adapters to GGUF for Ollama, run: ./convert_gguf.sh"


echo "=========================================================="
echo "🎉 buddy-cortex Pipeline Script Finished Successfully! 🎉"
echo "=========================================================="
