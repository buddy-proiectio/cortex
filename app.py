import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = FastAPI(
    title="Financial English-to-Korean Translation Engine",
    description="Fine-tuned NLLB-200 NMT Model serving translated news articles",
)

MODEL_DIR = "./nllb-financial-translator"
BASE_MODEL = "facebook/nllb-200-distilled-600M"

# Global model and tokenizer
model = None
tokenizer = None
device = "cpu"

@app.on_event("startup")
def load_model():
    global model, tokenizer, device
    
    # 1. Device selection
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"[*] Serving translation engine on device: {device}")

    # 2. Check if fine-tuned model exists
    if os.path.exists(MODEL_DIR) and os.path.exists(os.path.join(MODEL_DIR, "config.json")):
        load_path = MODEL_DIR
        print(f"[+] Found fine-tuned NLLB model at {load_path}. Loading...")
    else:
        load_path = BASE_MODEL
        print(f"[!] Fine-tuned model not found at {MODEL_DIR}. Falling back to base model: {load_path}")

    # 3. Load tokenizer and model
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            load_path,
            src_lang="eng_Latn",
            tgt_lang="kor_Kotn"
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(load_path)
        model.to(device)
        model.eval()
        print("[+] Model and tokenizer loaded successfully!")
    except Exception as e:
        print(f"[-] Critical error loading model: {e}")
        raise RuntimeError(f"Could not load translation model: {e}")

class TranslationRequest(BaseModel):
    text: str

class TranslationResponse(BaseModel):
    translated_text: str

@app.post("/translate", response_model=TranslationResponse)
def translate(request: TranslationRequest):
    global model, tokenizer, device
    
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")
    
    text = request.text.strip()
    if not text:
        return TranslationResponse(translated_text="")

    try:
        # Tokenize inputs
        inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate translation with target language forced
        # forced_bos_token_id matches 'kor_Kotn'
        kor_lang_id = tokenizer.convert_tokens_to_ids("kor_Kotn")
        
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=kor_lang_id,
                max_length=1024,
                num_beams=1,
                length_penalty=1.0,
            )
        
        # Decode output
        translated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True).strip()
        return TranslationResponse(translated_text=translated_text)
    
    except Exception as e:
        print(f"[-] Translation process error: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@app.get("/")
def health_check():
    global device
    model_status = "loaded" if model is not None else "not loaded"
    is_finetuned = os.path.exists(MODEL_DIR) and os.path.exists(os.path.join(MODEL_DIR, "config.json"))
    return {
        "status": "ok",
        "device": device,
        "model_status": model_status,
        "model_type": "fine-tuned" if is_finetuned else "base-nllb",
    }
