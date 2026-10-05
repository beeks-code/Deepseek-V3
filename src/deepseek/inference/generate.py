import torch


def generate(model, input_ids, max_new_tokens, context_size, temperature=0.8, top_k=40, eos_id=None):
    model.eval()
    model.reset_cache()
    input_ids = input_ids[:, -context_size:]

    with torch.no_grad():
        logits = model(input_ids, use_cache=True)

        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :]

            if top_k is not None:
                top_values, _ = torch.topk(next_logits, top_k)
                min_top_value = top_values[:, -1].unsqueeze(-1)
                next_logits = torch.where(
                    next_logits < min_top_value,
                    torch.full_like(next_logits, float("-inf")),
                    next_logits,
                )

            if temperature > 0:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

            if eos_id is not None and next_token.item() == eos_id:
                break

            input_ids = torch.cat([input_ids, next_token], dim=1)
            logits = model(next_token, use_cache=True)

    return input_ids


def text_to_token_ids(text, tokenizer):
    return torch.tensor(tokenizer.encode(text)).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.squeeze(0).tolist())
