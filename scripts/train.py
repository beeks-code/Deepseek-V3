import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deepseek.config import GPT_CONFIG_124M
from deepseek.data import create_dataloader
from deepseek.models import DeepSeekV3
from deepseek.training import train_one_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--out", default="deepseek_basic.pt")
    args = parser.parse_args()

    try:
        import tiktoken
    except ImportError as exc:
        raise SystemExit("Install tiktoken to use this script: pip install tiktoken") from exc

    text = Path(args.text_file).read_text(encoding="utf-8")
    tokenizer = tiktoken.get_encoding("gpt2")
    dataloader = create_dataloader(
        text,
        tokenizer,
        batch_size=args.batch_size,
        context_size=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepSeekV3(GPT_CONFIG_124M).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    for epoch in range(args.epochs):
        loss = train_one_epoch(model, dataloader, optimizer, device)
        print(f"epoch {epoch + 1}: loss {loss:.4f}")

    torch.save({"model_state_dict": model.state_dict(), "cfg": GPT_CONFIG_124M}, args.out)


if __name__ == "__main__":
    main()
