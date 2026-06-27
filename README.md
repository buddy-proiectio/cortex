# 🧠 buddy-cortex

`buddy-cortex` is a fine-tuning pipeline project for building an **English-to-Korean financial translation LLM** designed to translate global financial and macroeconomics news articles into a professional Korean news tone.

It is optimized for Apple Silicon (such as M-series Macs) using the Apple MLX framework (`mlx-lm`). All Python dependencies and virtual environments are managed cleanly and efficiently with `uv`.

---

## 📂 Directory Structure

```text
buddy-cortex/
├── data/                      # Directory for train.jsonl and valid.jsonl (auto-generated)
├── .python-version            # Recommended Python version for the project
├── pyproject.toml             # uv & project configuration (dependencies like mlx-lm)
├── uv.lock                    # Dependency lockfile
├── dataset.csv                # (User-provided) Source English and translation pairs
├── convert_dataset.py         # CSV -> Gemma-2 format JSONL converter script
├── train_pipeline.sh          # Shell script to automate data preparation and LoRA training
└── convert_gguf.sh            # Shell script to fuse FP16 base, convert to GGUF, and quantize
```

---

## 🛠️ Prerequisites

- **OS**: macOS (Apple Silicon M-series recommended)
- **Tools**: 
  - `uv` (a fast, modern Python package and environment manager)
    > If you do not have `uv` installed, install it using:
    > ```bash
    > curl -LsSf https://astral.sh/uv/install.sh | sh
    > ```
  - `cmake` (required for compiling `llama.cpp` quantization tools)
    > Install it using Homebrew:
    > ```bash
    > brew install cmake
    > ```
  - `ollama` (for running the quantized model locally)

---

## 🚀 Getting Started

### Step 1. Synchronize Project Dependencies

Run `uv sync` to create a virtual environment and install all required libraries automatically:
```bash
uv sync
```

### Step 2. Prepare the Dataset (`dataset.csv`)

Place your `dataset.csv` file in the root directory of the project.
- Required columns: `en_text` (English article text) and `ko_text` (Korean translation text).
  *(If column names do not match explicitly, the script will automatically fallback to the first two columns.)*

---

## 🏃 Execution Guide

### 1. Run the Unified Pipeline (`train_pipeline.sh`)

Automate the entire pipeline—from verifying prerequisites and preparing datasets to executing LoRA fine-tuning and merging (fusing) the weights.

**Dry-run (verify the environment setup and data prep without running the actual training):**
```bash
./train_pipeline.sh --dry-run
```

**Run actual LoRA Fine-Tuning and Merging:**
```bash
./train_pipeline.sh --iters 1000 --batch-size 1 --num-layers 16
```

* **Options**:
  * `-m, --model <str>`: Base model to use (default: `mlx-community/gemma-2-9b-it-4bit`)
  * `-d, --data <path>`: Directory to save converted JSONL files (default: `./data`)
  * `-l, --num-layers <int>`: Number of layers to apply LoRA (default: `16`)
  * `-b, --batch-size <int>`: Batch size for training (default: `1`)
  * `--grad-accumulation-steps <int>`: Number of steps to accumulate gradients (default: `4`)
  * `-i, --iters <int>`: Number of training iterations (default: `1000`)
  * `-a, --adapter-path <path>`: Directory to save adapters (default: `adapters`)
  * `-s, --save-path <path>`: Directory to save the fused model (default: `fused_model`)
  * `--max-seq-length <int>`: Maximum sequence length (default: `2048`)
  * `--val-batches <int>`: Number of validation batches (default: `10`, use `-1` for all)
  * `--no-grad-checkpoint`: Disable gradient checkpointing to speed up training if memory is sufficient

---

### 2. Run Standalone Data Conversion (`convert_dataset.py`)

If you want to split and format the dataset into Gemma-2 compatible `train.jsonl` and `valid.jsonl` files without running the training pipeline:
```bash
uv run python convert_dataset.py --val-ratio 0.1 --seed 42
```
* Once completed successfully, `train.jsonl` and `valid.jsonl` will be generated in `./data/`.

---

## 🦙 GGUF Conversion & Ollama Deployment

To run your fine-tuned model in Ollama, you must convert the trained weights (LoRA adapters) into the GGUF format and quantize them. A dedicated script, `convert_gguf.sh`, is provided to automate this entire pipeline.

### Hugging Face Authentication & Gated Model Access (Crucial)
Since we are using **Gemma 2 9B Instruct** as the base model, which is a **gated model** on Hugging Face, you must set up authentication before converting the model. If you bypass this, the fusion script will throw a `403 Forbidden` error.

1. **Accept the License Agreement**:
   Visit the [google/gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it) page on Hugging Face, log in, and accept the license terms ("Agree and access repository").
2. **Log In via CLI**:
   Note that the legacy `huggingface-cli login` tool is deprecated. Use the modern Hugging Face CLI command to authenticate:
   ```bash
   hf auth login
   ```
   *Note: If the CLI does not pass the credentials correctly to the python environment, you can set the token directly as an environment variable:*
   ```bash
   export HF_TOKEN="your_huggingface_access_token"
   ```

### 1. Run GGUF Conversion Script (`convert_gguf.sh`)
Execute the conversion script to automatically fuse adapters with the unquantized FP16 base model, build `llama.cpp`, convert to GGUF, quantize, and clean up temporary files:
```bash
./convert_gguf.sh
```
* **Options**:
  * `-m, --model <str>`: Base unquantized Hugging Face model (default: `google/gemma-2-9b-it`)
  * `-a, --adapter-path <path>`: Directory containing adapters (default: `adapters`)
  * `-q, --quantization <str>`: Quantization format (default: `Q4_K_M`)
  * `-o, --output <path>`: Final GGUF path (default: `fused_model_q4_k_m.gguf`)
  * `-n, --name <str>`: Ollama model registration name (default: `buddy-cortex`)

### 2. Register & Run in Ollama
Once `convert_gguf.sh` completes, it generates a `Modelfile` in the root directory. Register and run the model in Ollama:
```bash
# 1. Register the model in Ollama
ollama create buddy-cortex -f Modelfile

# 2. Run the model
ollama run buddy-cortex
```
