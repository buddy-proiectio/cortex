# 🧠 buddy-cortex

`buddy-cortex` is a translation engine and fine-tuning workspace designed to build and serve high-quality **English-to-Korean financial/macroeconomics news translation models**.

It currently supports:
1. **Fine-Tuning & Quantization Pipeline** using the MLX framework (`mlx-lm`) and EXAONE 3.5.
2. **NLLB-200 dist-600M NMT Serving Engine** running a FastAPI server on port `8000`.

---

## 📂 Directory Structure

```text
buddy-cortex/
├── data/                      # Converted JSONL datasets for MLX training
├── .python-version            # Python runtime version
├── pyproject.toml             # uv dependencies
├── uv.lock                    # Locked dependencies
├── dataset.csv                # Financial news translation dataset (en_text, ko_text)
├── app.py                     # FastAPI server serving NLLB-200 translation (port 8000)
├── train_nmt.py               # Seq2Seq fine-tuning script for NLLB-200
├── convert_dataset.py         # Dataset converter for EXAONE/MLX formats
├── train_pipeline.sh          # Auto-trainer script for EXAONE/MLX LoRA
└── convert_gguf.sh            # FP16 merger, GGUF converter, and Ollama registrar
```

---

## 🚀 Serving Engine (NLLB-200 NMT)

The translation pipeline in `buddy-core` depends on the NMT server running on `http://127.0.0.1:8000`.

### 1. Launching the Translation Server
Run the Uvicorn FastAPI server manually or let `buddy-core` manage its lifecycle dynamically:
```bash
# In buddy-cortex directory
.venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
```
* The server will automatically load the local fine-tuned model located at `./nllb-financial-translator`, falling back to the base model `facebook/nllb-200-distilled-600M` if not found.

### 2. API Endpoints
* **Health Check**: `GET http://127.0.0.1:8000/`
  * Returns: `{"status": "ok", "device": "mps", "model_type": "fine-tuned"}`
* **Translate**: `POST http://127.0.0.1:8000/translate`
  * Payload: `{"text": "English text here..."}`
  * Response: `{"translated_text": "한국어 번역본..."}`

---

## 📈 Incremental Fine-Tuning Guide (Warm-Starting)

To continuously improve translation quality as you gather more high-quality translation data:

### Step 1. Append New Translation Pairs
Open `dataset.csv` and append your new translation pairs to the end of the file, maintaining the column structure:
```csv
en_text,ko_text
"Original English news sentence...","전문 금융 번역 한국어..."
```

### Step 2. Warm-Start Training (NLLB-200 NMT)
Instead of training from scratch using the raw base model, you can continue training from your already fine-tuned checkpoints (`./nllb-financial-translator`) to preserve prior learning.

1. Open `train_nmt.py`.
2. Locate the model configuration at line 111:
   ```python
   # Modify this line to point to the local directory
   model_name = "./nllb-financial-translator"
   ```
3. Run the training script:
   ```bash
   .venv/bin/python train_nmt.py --epochs 3 --learning-rate 2e-5
   ```
   * **Tip**: For warm-starts, use a lower learning rate (e.g. `2e-5` or `1e-5`) and fewer epochs (`2` or `3`) to prevent overriding previously learned weights (catastrophic forgetting).

### Step 3. Warm-Start Training (EXAONE-3.5 MLX)
To incrementally train the EXAONE-3.5 model using the MLX framework:
1. Append to `dataset.csv`.
2. Re-convert your data using:
   ```bash
   uv run python convert_dataset.py
   ```
3. Run `train_pipeline.sh` pointing to the previous adapters or model folder:
   ```bash
   ./train_pipeline.sh --iters 500 --learning-rate 1e-6
   ```
