import argparse
from pathlib import Path
from mlx.utils import tree_unflatten
from mlx_lm.utils import load, save

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse fine-tuned adapters into the base model with trust-remote-code enabled."
    )
    parser.add_argument(
        "--model",
        default="mlx_model",
        help="The path to the local model directory or Hugging Face repo.",
    )
    parser.add_argument(
        "--save-path",
        default="fused_model",
        help="The path to save the fused model.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="adapters",
        help="Path to the trained adapter weights and config.",
    )
    return parser.parse_args()

def main() -> None:
    print("[*] Loading pretrained model and adapters with trust_remote_code=True...")
    args = parse_arguments()

    # Pass trust_remote_code=True and lazy=True to avoid memory exhaustion (OOM)
    model, tokenizer, config = load(
        args.model,
        adapter_path=args.adapter_path,
        return_config=True,
        tokenizer_config={"trust_remote_code": True},
        model_config={"trust_remote_code": True},
        lazy=True
    )

    fused_linears = [
        (n, m.fuse(dequantize=False))
        for n, m in model.named_modules()
        if hasattr(m, "fuse")
    ]

    if fused_linears:
        model.update_modules(tree_unflatten(fused_linears))

    save_path = Path(args.save_path)
    save(
        save_path,
        args.model,
        model,
        tokenizer,
        config,
        donate_model=False,
    )
    print(f"[+] Fused model successfully saved to: {args.save_path}")

if __name__ == "__main__":
    main()
