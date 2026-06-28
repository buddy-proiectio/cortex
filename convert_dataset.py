"""
convert_dataset.py

This script processes the source 'dataset.csv' file and formats the English and
Korean financial news articles into EXAONE-3.5-7.8B-Instruct compatible prompt structures. It then
shuffles and splits the formatted data into 'train.jsonl' and 'valid.jsonl' files
inside the target directory (default: './data') to be used for MLX fine-tuning.

Requirements:
- Python 3.6+ (No external dependencies like pandas are strictly required,
  using standard libraries for maximum compatibility and speed).
"""

import argparse
import csv
import json
import os
import random
import sys


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Convert and split CSV financial news dataset into EXAONE-3.5-7.8B-Instruct formatted JSONL files for MLX."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="dataset.csv",
        help="Path to the input CSV file (default: dataset.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory to save train.jsonl and valid.jsonl (default: data)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Ratio of validation dataset (default: 0.1 for 10%%)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling of the dataset before splitting",
    )
    return parser.parse_args()


def format_exaone_prompt(en_text, ko_text):
    """
    Formats the raw English and Korean articles into the EXAONE-3.5 special token template,
    using a sophisticated, finance-specialized system prompt, and using raw text in the user/assistant slots.
    """
    system_prompt = (
        "You are a professional financial translator specializing in translating global financial and macroeconomic news articles. "
        "Translate the given English financial news into professional, natural, and accurate Korean news tone. "
        "Ensure all financial terms, indicators, and names are translated correctly according to standard financial terminology. "
        "Maintain exact numerical values, currencies, percentages, and dates without any modification. "
        "Do not add any conversational remarks, introductions, or explanations; output only the translated text."
    )

    # Ensure any windows-style newlines or extra whitespaces are standardized
    en_text = en_text.strip()
    ko_text = ko_text.strip()

    prompt = (
        f"[|system|]{system_prompt}[|endofturn|]\n"
        f"[|user|]{en_text}\n"
        f"[|assistant|]{ko_text}[|endofturn|]"
    )
    return {"text": prompt}


def main():
    args = parse_arguments()

    if not os.path.exists(args.input):
        print(f"[-] Error: Input file '{args.input}' not found.", file=sys.stderr)
        print(
            "    Please make sure the dataset is placed at the specified path or run the script with --input.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[*] Reading dataset from: {args.input}")

    records = []

    # We read with utf-8-sig to automatically handle Byte Order Mark (BOM) if exported from Excel/Google Sheets
    with open(args.input, mode="r", encoding="utf-8-sig") as csv_file:
        # We can handle both comma-separated and semicolon-separated CSVs dynamically
        dialect = (
            csv.Sniffer().sniff(csv_file.read(2048))
            if os.path.getsize(args.input) > 10
            else None
        )
        csv_file.seek(0)

        reader = (
            csv.DictReader(csv_file, dialect=dialect)
            if dialect
            else csv.DictReader(csv_file)
        )

        # Verify the columns exist
        headers = reader.fieldnames
        if not headers:
            print(
                "[-] Error: The CSV file is empty or headers could not be parsed.",
                file=sys.stderr,
            )
            sys.exit(1)

        en_col = None
        ko_col = None

        # Look for matching columns case-insensitively or by common patterns
        for header in headers:
            h_lower = header.lower().strip()
            if h_lower == "en_text" or "english" in h_lower or "en" == h_lower:
                en_col = header
            elif h_lower == "ko_text" or "korean" in h_lower or "ko" == h_lower:
                ko_col = header

        # Fallback to column index 0 and 1 if standard column names aren't matched
        if not en_col or not ko_col:
            if len(headers) >= 2:
                en_col = headers[0]
                ko_col = headers[1]
                print(
                    "[!] Warning: Columns 'en_text' and 'ko_text' not explicitly found."
                )
                print(
                    f"    Mapping Column 0 ('{en_col}') to English and Column 1 ('{ko_col}') to Korean."
                )
            else:
                print(
                    f"[-] Error: CSV must contain at least 2 columns (English and Korean text). Available columns: {headers}",
                    file=sys.stderr,
                )
                sys.exit(1)

        # Read and format each row
        for line_num, row in enumerate(reader, start=2):
            en_val = row.get(en_col)
            ko_val = row.get(ko_col)

            if not en_val or not ko_val or not en_val.strip() or not ko_val.strip():
                # Skip empty or incomplete rows but log a warning
                continue

            formatted = format_exaone_prompt(en_val, ko_val)
            records.append(formatted)

    total_records = len(records)
    print(f"[+] Successfully loaded and formatted {total_records} rows from CSV.")

    if total_records == 0:
        print("[-] Error: No valid rows found in the CSV dataset.", file=sys.stderr)
        sys.exit(1)

    # Shuffle the records for uniform distribution unless disabled
    if not args.no_shuffle:
        print(f"[*] Shuffling dataset with seed {args.seed}...")
        random.seed(args.seed)
        random.shuffle(records)

    # Split into train and validation sets
    val_count = int(total_records * args.val_ratio)
    if val_count == 0 and args.val_ratio > 0 and total_records > 1:
        val_count = 1  # Ensure at least 1 validation sample if ratio > 0

    valid_records = records[:val_count]
    train_records = records[val_count:]

    print("[+] Dataset Split Summary:")
    print(f"    - Total: {total_records}")
    print(
        f"    - Training: {len(train_records)} ({((len(train_records) / total_records) * 100):.1f}%)"
    )
    print(
        f"    - Validation: {len(valid_records)} ({((len(valid_records) / total_records) * 100):.1f}%)"
    )

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, "train.jsonl")
    valid_path = os.path.join(args.output_dir, "valid.jsonl")

    # Save training set
    with open(train_path, mode="w", encoding="utf-8") as f:
        for item in train_records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Save validation set
    with open(valid_path, mode="w", encoding="utf-8") as f:
        for item in valid_records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[+] Saved training dataset to: {train_path}")
    print(f"[+] Saved validation dataset to: {valid_path}")
    print("[+] Conversion completed successfully!")


if __name__ == "__main__":
    main()
