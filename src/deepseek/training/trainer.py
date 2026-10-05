import torch

from .loss import deepseek_loss


def train_one_epoch(model, dataloader, optimizer, device, mtp_weight=0.3):
    model.train()
    total_loss = 0.0

    for input_ids, targets in dataloader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        output = model(input_ids, return_mtp=True)
        loss = deepseek_loss(output, targets, mtp_weight=mtp_weight)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(1, len(dataloader))


@torch.no_grad()
def evaluate(model, dataloader, device, mtp_weight=0.3):
    model.eval()
    total_loss = 0.0

    for input_ids, targets in dataloader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        output = model(input_ids, return_mtp=True)
        total_loss += deepseek_loss(output, targets, mtp_weight=mtp_weight).item()

    return total_loss / max(1, len(dataloader))
