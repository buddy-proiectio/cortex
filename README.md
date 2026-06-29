# 🧠 buddy-cortex

`buddy-cortex` is a translation engine and fine-tuning workspace designed to build and serve high-quality **English-to-Korean financial and macroeconomic news translation models** based on NLLB-200.

---

## 🚀 Features

1. **NLLB-200 Fine-Tuning Pipeline**: Easily train a Seq2Seq model (`facebook/nllb-200-distilled-600M`) on your custom English-Korean translation datasets.
2. **FastAPI Serving Engine**: Serve translations locally on port `8000` with automatic fallback to the base model if a fine-tuned model is not yet available.
3. **Modern Python Stack**: Packaged and managed using [uv](https://astral.sh/uv) for fast, reliable, and reproducible dependency management.

---

## 📂 Directory Structure

```text
buddy-cortex/
├── .python-version            # Target Python runtime version
├── pyproject.toml             # Project metadata and dependencies managed by uv
├── uv.lock                    # Dependency lockfile
├── dataset.csv                # Financial news translation dataset (en_text, ko_text)
├── app.py                     # FastAPI server for NLLB-200 translation
├── train_nmt.py               # Fine-tuning script for NLLB-200 Seq2Seq model
├── train_pipeline.sh          # Orchestration script to run dependencies sync and training
└── nllb-financial-translator/ # Directory containing the fine-tuned model (created after training)
```

---

## ⚙️ Setup & Installation

This project uses `uv` for python environment and dependency management.

1. **Install uv** (if you haven't already):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Sync Dependencies**:
   ```bash
   uv sync
   ```
   This will automatically create a virtual environment (`.venv`) and install all required libraries.

---

## 📈 Fine-Tuning the NMT Model

You can fine-tune the model using the orchestration script `train_pipeline.sh` or run the Python training script directly.

### Option 1. Using the pipeline script (Recommended)
Run the auto-trainer script:
```bash
chmod +x train_pipeline.sh
./train_pipeline.sh --epochs 5 --batch-size 2 --lr 5e-5
```

**Available Options:**
* `-e, --epochs <int>`: Number of training epochs (default: `5`)
* `-b, --batch-size <int>`: Batch size per device (default: `1`)
* `-r, --lr <float>`: Learning rate (default: `5e-5`)
* `-i, --input <path>`: Dataset CSV file (default: `dataset.csv`)
* `-o, --output <path>`: Directory to save the model (default: `./nllb-financial-translator`)
* `--cpu`: Force CPU training
* `--bf16`: Enable bfloat16 mixed precision training (recommended for MPS/CUDA)
* `--fp16`: Enable fp16 mixed precision training

### Option 2. Running Python training directly
```bash
uv run python train_nmt.py --input dataset.csv --output-dir ./nllb-financial-translator --epochs 5
```

---

## 🚀 Serving the Translation API

The translation server serves NLLB-200 translation via HTTP.

### 1. Launching the Server
Run the FastAPI application using Uvicorn:
```bash
uv run uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```
* **Auto-fallback**: The server checks for a fine-tuned model in `./nllb-financial-translator`. If it exists, it loads it. Otherwise, it falls back to the base `facebook/nllb-200-distilled-600M` model.

### 2. API Endpoints

#### **Health Check**
* **Request**: `GET http://127.0.0.1:8000/`
* **Response**:
  ```json
  {
    "status": "ok",
    "device": "mps",
    "model_status": "loaded",
    "model_type": "fine-tuned" 
  }
  ```
  *(Note: `model_type` will be `"base-nllb"` if local weights are not found)*

#### **Translate Text**
* **Request**: `POST http://127.0.0.1:8000/translate`
* **Payload**:
  ```json
  {
    "text": "The Federal Reserve left interest rates unchanged at its meeting today."
  }
  ```
* **Response**:
  ```json
  {
    "translated_text": "연방준비제도는 오늘 회의에서 금리를 동결했습니다."
  }
  ```

---

## ✍️ Dataset Management

To expand or improve the translation quality, add translation pairs directly to the `dataset.csv` file:
```csv
en_text,ko_text
"Original English news sentence...","전문 금융 번역 한국어..."
```
Ensure you keep the headers `en_text` and `ko_text`. Run the fine-tuning pipeline after adding new dataset entries.
