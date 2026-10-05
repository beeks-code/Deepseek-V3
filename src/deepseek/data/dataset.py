import torch
from torch.utils.data import DataLoader, Dataset
class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, context_size, stride):
        self.input_ids = []
        self.target_ids = []
        tokens = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

        for i in range(0, len(tokens) - context_size, stride):
            input_tokens = tokens[i:i + context_size]
            target_tokens = tokens[i + 1:i + context_size + 1]
            self.input_ids.append(torch.tensor(input_tokens, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_tokens, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]

def create_dataloader(
    text,
    tokenizer,
    batch_size=4,
    context_size=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=3,
):
    dataset = GPTDataset(text, tokenizer, context_size, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
