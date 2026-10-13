import torch.nn.functional as F

from .bopomofo_utils import remove_bopomofo_padding


def ctc_loss(logits, batch_idx, bopomofo_ids_padded, bopomofo_lengths,
             cnn_time_lengths, criterion, device):
    """
    Compute CTC loss from the model output for one mini-batch.

    logits:
        model output, shape = (batch, output_time, vocab_size)

    bopomofo_ids_padded / bopomofo_lengths:
        correct Bopomofo ID sequences; padding is removed before CTCLoss

    cnn_time_lengths:
        valid output-time length of each speech sample after CNN downsampling
    """
    log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)

    correct_bopomofo_ids, correct_bopomofo_lengths = remove_bopomofo_padding(
        batch_idx, bopomofo_ids_padded, bopomofo_lengths, device
    )

    batch_time_lengths = cnn_time_lengths[batch_idx]

    return criterion(
        log_probs,
        correct_bopomofo_ids,
        batch_time_lengths,
        correct_bopomofo_lengths
    )
