#!/usr/bin/env zsh
#
# convert_gguf.sh
#
# Automates the process of converting MLX LoRA adapters into a GGUF model for Ollama.
# Handles FP16 fusing, GGUF conversion, quantization, temporary file cleanup, and Modelfile generation.

set -e

# Default configurations
BASE_MODEL="google/gemma-2-9b-it"
ADAPTER_PATH="adapters"
TEMP_FUSED_DIR="fused_model_fp16"
TEMP_GGUF_FP16="fused_model_fp16.gguf"
FINAL_GGUF="fused_model_q4_k_m.gguf"
QUANTIZATION="Q4_K_M"
OLLAMA_MODEL_NAME="buddy-cortex"

# Helper function to print usage
print_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -m, --model <str>          Base unquantized HF model (default: $BASE_MODEL)"
    echo "  -a, --adapter-path <path>  Directory containing adapters (default: $ADAPTER_PATH)"
    echo "  -q, --quantization <str>   GGUF quantization format (default: $QUANTIZATION)"
    echo "  -o, --output <path>        Output GGUF file name (default: $FINAL_GGUF)"
    echo "  -n, --name <str>           Ollama model name to register (default: $OLLAMA_MODEL_NAME)"
    echo "  -h, --help                 Show this help message"
    echo ""
}

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -m|--model) BASE_MODEL="$2"; shift ;;
        -a|--adapter-path) ADAPTER_PATH="$2"; shift ;;
        -q|--quantization) QUANTIZATION="$2"; shift ;;
        -o|--output) FINAL_GGUF="$2"; shift ;;
        -n|--name) OLLAMA_MODEL_NAME="$2"; shift ;;
        -h|--help) print_usage; exit 0 ;;
        *) echo "[-] Unknown parameter passed: $1"; print_usage; exit 1 ;;
    esac
    shift
done

echo "=========================================================="
echo "    🦙 GGUF Conversion & Ollama Import Pipeline 🦙"
echo "=========================================================="
echo "[*] Base HF Model:          $BASE_MODEL"
echo "[*] Adapter Path:           $ADAPTER_PATH"
echo "[*] Target Quantization:    $QUANTIZATION"
echo "[*] Output GGUF:            $FINAL_GGUF"
echo "[*] Ollama Model Name:      $OLLAMA_MODEL_NAME"
echo "=========================================================="

# 1. Verify environment
if ! command -v uv &>/dev/null; then
    echo "[-] Error: 'uv' command not found."
    echo "    Please install uv (https://astral.sh/uv) and try again."
    exit 1
fi

if [ ! -d "$ADAPTER_PATH" ]; then
    echo "[-] Error: Adapter directory '$ADAPTER_PATH' not found."
    echo "    Please run training first (e.g. ./train_pipeline.sh)."
    exit 1
fi

# 2. Step 1: Fuse adapters into unquantized base model
echo "[*] Step 1: Fusing adapters with unquantized base model ($BASE_MODEL) in FP16/BF16..."
echo "    (This might download the base model files if not already cached)"
uv run python -m mlx_lm fuse \
  --model "$BASE_MODEL" \
  --adapter-path "$ADAPTER_PATH" \
  --save-path "$TEMP_FUSED_DIR"

# Download missing tokenizer.model since mlx_lm fuse doesn't copy it, but llama.cpp requires it for Gemma/Gemma-2
if [ ! -f "$TEMP_FUSED_DIR/tokenizer.model" ]; then
    echo "[*] Downloading tokenizer.model from Hugging Face for $BASE_MODEL..."
    uv run python -c "
from huggingface_hub import hf_hub_download
import sys
try:
    hf_hub_download(repo_id='$BASE_MODEL', filename='tokenizer.model', local_dir='$TEMP_FUSED_DIR')
    print('[+] Successfully downloaded tokenizer.model')
except Exception as e:
    print(f'[-] Warning: Could not download tokenizer.model: {e}', file=sys.stderr)
"
fi


# 3. Setup llama.cpp
if [ ! -d "llama.cpp" ]; then
    echo "[*] Step 2: 'llama.cpp' not found. Cloning repository..."
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git
fi

echo "[*] Step 3: Installing llama.cpp python dependencies..."
uv pip install --index-strategy unsafe-best-match -r llama.cpp/requirements.txt

# 4. Convert HF FP16 to GGUF FP16
echo "[*] Step 4: Converting fused HF model to FP16 GGUF..."
uv run python llama.cpp/convert_hf_to_gguf.py "$TEMP_FUSED_DIR" \
  --outfile "$TEMP_GGUF_FP16" \
  --outtype f16

# 5. Build quantize tool & Quantize
if [ ! -f "llama.cpp/build/bin/llama-quantize" ]; then
    echo "[*] Step 5: Compiling llama-quantize tool using CMake..."
    # Verify cmake is installed
    if ! command -v cmake &>/dev/null; then
        echo "[-] Error: 'cmake' is required to compile llama.cpp."
        echo "    Please install it (e.g. 'brew install cmake') and try again."
        exit 1
    fi
    cd llama.cpp
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release --target llama-quantize
    cd ..
fi

echo "[*] Step 6: Quantizing GGUF to $QUANTIZATION..."
./llama.cpp/build/bin/llama-quantize "$TEMP_GGUF_FP16" "$FINAL_GGUF" "$QUANTIZATION"


# 6. Create Ollama Modelfile
echo "[*] Step 7: Generating Modelfile for Ollama..."
cat << 'EOF' > Modelfile
FROM ./fused_model_q4_k_m.gguf

# Gemma-2 chat template (User / Model alternation)
TEMPLATE """{{ if .Prompt }}<start_of_turn>user
{{ .Prompt }}<end_of_turn>
{{ end }}<start_of_turn>model
{{ .Response }}<end_of_turn>
"""

PARAMETER stop "<end_of_turn>"
PARAMETER stop "<eos>"
PARAMETER temperature 0.7
EOF
echo "[+] Modelfile generated successfully."

# 7. Clean up temporary files to save disk space
echo "[*] Step 8: Cleaning up temporary high-precision files..."
if [ -d "$TEMP_FUSED_DIR" ]; then
    rm -rf "$TEMP_FUSED_DIR"
    echo "[+] Removed temporary directory: $TEMP_FUSED_DIR"
fi
if [ -f "$TEMP_GGUF_FP16" ]; then
    rm -f "$TEMP_GGUF_FP16"
    echo "[+] Removed temporary file: $TEMP_GGUF_FP16"
fi

echo "=========================================================="
echo "🎉 GGUF Conversion Completed Successfully! 🎉"
echo "=========================================================="
echo "[*] Final Quantized GGUF:  $FINAL_GGUF"
echo "[*] Register & run in Ollama by executing:"
echo "    ollama create $OLLAMA_MODEL_NAME -f Modelfile"
echo "    ollama run $OLLAMA_MODEL_NAME"
echo "=========================================================="
