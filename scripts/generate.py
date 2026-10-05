import argparse
import sys
from pathlib import Path

import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deepseek.config import GPT_CONFIG_124M
from deepseek.inference import generate, text_to_token_ids, token_ids_to_text
from deepseek.models import DeepSeekV3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Every effort moves you")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    try:
        import tiktoken
    except ImportError as exc:
        raise SystemExit("Please try to install tiktoken") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")
    model = DeepSeekV3(GPT_CONFIG_124M).to(device)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)

    input_ids = text_to_token_ids(args.prompt, tokenizer).to(device)
    output_ids = generate(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        context_size=GPT_CONFIG_124M["context_length"],
        eos_id=50256,
    )
    print(token_ids_to_text(output_ids.cpu(), tokenizer))


if __name__ == "__main__":
    main()
