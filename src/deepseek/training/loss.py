
import torch.nn.functional as F


def cross_entropy_loss(logits, targets):
    """
    Standard next-token cross-entropy loss.

    logits:  (batch, sequence, vocab)
    targets: (batch, sequence)
    """
    logits = logits.reshape(-1, logits.size(-1))
    targets = targets.reshape(-1)

    return F.cross_entropy(logits, targets)


def deepseek_loss(output, targets, mtp_weight=0.3):
    """
    Main language-model loss + auxiliary multi-token prediction losses.

    The main head predicts the next token:
        x[t] -> token[t + 1]

    MTP head k predicts:
        x[t] -> token[t + k + 1]
    """

    # Model only returns the main prediction head.
    if not isinstance(output, tuple):
        return cross_entropy_loss(output, targets)

    logits, mtp_logits = output

    # Standard next-token prediction loss.
    loss = cross_entropy_loss(logits, targets)

    # Multi-token prediction losses.
    for step, step_logits in enumerate(mtp_logits, start=1):
        if targets.size(1) <= step:
            continue

        # Remove the final `step` positions because there are
        # no targets available that far into the sequence.
        predictions = step_logits[:, :-step, :]
        expected = targets[:, step:]

        loss = loss + mtp_weight * cross_entropy_loss(
            predictions,
            expected,
        )

    return loss

