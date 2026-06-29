import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune NLLB-200 on Financial News Dataset")
    parser.add_argument(
        "--input",
        type=str,
        default="dataset.csv",
        help="Path to the input CSV file containing en_text and ko_text",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./nllb-financial-translator",
        help="Directory to save the fine-tuned model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate for fine-tuning",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU training even if MPS/CUDA is available",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Enable bfloat16 mixed precision training (recommended for MPS/CUDA)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable fp16 mixed precision training",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 0. CPU Thread optimization
    torch.set_num_threads(4)
    
    # 1. Device check
    device = "cpu"
    if args.cpu:
        print("[*] Forcing CPU training as requested.")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("[+] Apple Silicon GPU (MPS) is available! Using MPS.")
    elif torch.cuda.is_available():
        device = "cuda"
        print("[+] CUDA is available! Using CUDA.")
    else:
        print("[*] Using CPU (training may be slow).")

    # 2. Load dataset
    if not os.path.exists(args.input):
        print(f"[-] Error: Input file '{args.input}' not found.")
        return

    print(f"[*] Loading dataset from: {args.input}")
    df = pd.read_csv(args.input)
    
    # Check column names
    en_col = [c for c in df.columns if "en" in c.lower()][0]
    ko_col = [c for c in df.columns if "ko" in c.lower()][0]
    print(f"[+] Found column mapping: '{en_col}' -> '{ko_col}'")

    # Remove empty rows
    df = df.dropna(subset=[en_col, ko_col])
    print(f"[+] Loaded {len(df)} non-empty rows.")

    # Split into train/validation (90/10)
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    print(f"[+] Split sizes: Training={len(train_df)}, Validation={len(val_df)}")

    # Convert to Hugging Face Dataset format
    raw_datasets = DatasetDict({
        "train": Dataset.from_pandas(train_df[[en_col, ko_col]]),
        "validation": Dataset.from_pandas(val_df[[en_col, ko_col]])
    })

    # 3. Load Model and Tokenizer
    model_name = "facebook/nllb-200-distilled-600M"
    print(f"[*] Downloading/Loading base model and tokenizer: {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        src_lang="eng_Latn",
        tgt_lang="kor_Kotn"
    )
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)

    # 4. Preprocessing function
    def preprocess_function(examples):
        inputs = [ex.strip() for ex in examples[en_col]]
        targets = [ex.strip() for ex in examples[ko_col]]
        
        # Tokenize source
        model_inputs = tokenizer(inputs, max_length=256, truncation=True)
        
        # Tokenize target (using text_target parameter)
        labels = tokenizer(text_target=targets, max_length=256, truncation=True)
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("[*] Preprocessing datasets...")
    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names
    )

    # Data Collator for padding dynamically
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # 5. Training Arguments
    print("[*] Setting up training arguments...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        fp16=args.fp16,
        bf16=args.bf16,
        use_cpu=(device == "cpu"),
        dataloader_pin_memory=False, # Avoid warnings & slowdowns on MPS
        dataloader_num_workers=0,    # Avoid Loky/multiprocessing lockups
        logging_steps=20,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        report_to="none", # Disable logging to wandb or other platforms
    )

    # 6. Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # 7. Train and Save
    print("[*] Starting translation model fine-tuning...")
    trainer.train()
    
    print(f"[*] Saving best model and tokenizer to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("[+] Fine-tuning completed successfully!")

if __name__ == "__main__":
    main()
