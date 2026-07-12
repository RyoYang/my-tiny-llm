import argparse

import torch

from .checkpoints import load_model
from .inference import generate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PyTorch tiny Qwen3 model")
    parser.add_argument("--model", default="qwen3-0.6b")
    parser.add_argument(
        "--prompt",
        default="Give me a short introduction to large language models.",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model, tokenizer = load_model(
        args.model,
        device=device,
        dtype=dtype,
        local_files_only=args.local_files_only,
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": args.prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,
    )
    text = generate(
        model,
        tokenizer,
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    print(text)


if __name__ == "__main__":
    main()
