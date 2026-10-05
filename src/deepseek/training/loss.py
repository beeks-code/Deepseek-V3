import torch.nn.functional as F


def cross_entropy_loss(logits, targets):
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten())


def deepseek_loss(output, targets, mtp_weight=0.3):
    if not isinstance(output, tuple):
        return cross_entropy_loss(output, targets)

    logits, mtp_logits = output
    loss = cross_entropy_loss(logits, targets)

    for step, step_logits in enumerate(mtp_logits, start=1):
        if targets.size(1) <= step:
            continue
        pred = step_logits[:, :-step, :]
        target = targets[:, step:]
        loss = loss + mtp_weight * cross_entropy_loss(pred, target)

    return loss
