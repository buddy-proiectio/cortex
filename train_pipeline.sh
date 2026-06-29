#!/usr/bin/env zsh
#
# train_pipeline.sh
#
# Automates the training process for the NLLB-200 English-to-Korean translation model.
#

set -e

# Default configurations
EPOCHS=5
BATCH_SIZE=1
LEARNING_RATE=5e-5
INPUT_CSV="dataset.csv"
OUTPUT_DIR="./nllb-financial-translator"
CPU=false
BF16=false
FP16=false

print_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -e, --epochs <int>         Number of training epochs (default: $EPOCHS)"
    echo "  -b, --batch-size <int>     Batch size for training (default: $BATCH_SIZE)"
    echo "  -r, --lr <float>           Learning rate (default: $LEARNING_RATE)"
    echo "  -i, --input <path>         Input CSV dataset (default: $INPUT_CSV)"
    echo "  -o, --output <path>        Output model directory (default: $OUTPUT_DIR)"
    echo "  --cpu                      Force CPU training (ignores GPU)"
    echo "  --bf16                     Enable bfloat16 mixed precision"
    echo "  --fp16                     Enable fp16 mixed precision"
    echo "  -h, --help                 Show this help message"
    echo ""
}

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -e|--epochs) EPOCHS="$2"; shift ;;
        -b|--batch-size) BATCH_SIZE="$2"; shift ;;
        -r|--lr) LEARNING_RATE="$2"; shift ;;
        -i|--input) INPUT_CSV="$2"; shift ;;
        -o|--output) OUTPUT_DIR="$2"; shift ;;
        --cpu) CPU=true ;;
        --bf16) BF16=true ;;
        --fp16) FP16=true ;;
        -h|--help) print_usage; exit 0 ;;
        *) echo "[-] Unknown parameter passed: $1"; print_usage; exit 1 ;;
    esac
    shift
done

echo "=========================================================="
echo "    🚀 buddy-cortex NMT Fine-Tuning Pipeline 🚀"
echo "=========================================================="
echo "[*] Input Dataset:     $INPUT_CSV"
echo "[*] Output Directory:  $OUTPUT_DIR"
echo "[*] Epochs:            $EPOCHS"
echo "[*] Batch Size:        $BATCH_SIZE"
echo "[*] Learning Rate:     $LEARNING_RATE"
echo "=========================================================="

# 1. Prerequisite verification
if ! command -v uv &>/dev/null; then
    echo "[-] Error: 'uv' command not found."
    echo "    Please install uv (https://astral.sh/uv) and try again."
    exit 1
fi

echo "[*] Synchronizing dependencies using uv..."
uv sync

# 2. Dataset check
if [ ! -f "$INPUT_CSV" ]; then
    echo "[-] Error: Dataset file '$INPUT_CSV' not found."
    echo "    Please make sure dataset.csv exists in this directory."
    exit 1
fi

# 3. Start training
EXTRA_ARGS=()
if [ "$CPU" = true ]; then
    EXTRA_ARGS+=(--cpu)
fi
if [ "$BF16" = true ]; then
    EXTRA_ARGS+=(--bf16)
fi
if [ "$FP16" = true ]; then
    EXTRA_ARGS+=(--fp16)
fi

echo "[*] Initiating NLLB fine-tuning script..."
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 uv run python train_nmt.py \
    --input "$INPUT_CSV" \
    --output-dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    "${EXTRA_ARGS[@]}"

echo "=========================================================="
echo "🎉 buddy-cortex Translation Fine-Tuning Complete! 🎉"
echo "=========================================================="
echo "[*] Fine-tuned model saved in: $OUTPUT_DIR"
echo "[*] To run the translation FastAPI server, execute:"
echo "    uv run uvicorn app:app --reload --host 127.0.0.1 --port 8000"
echo "=========================================================="
