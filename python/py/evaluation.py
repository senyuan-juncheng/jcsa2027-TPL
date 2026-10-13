import torch

from .loss import ctc_loss
from .metrics import strict_metrics, tw_metrics


@torch.no_grad()
def predict_bopomofo_ids(indices, model, X, cnn_time_lengths, blank_id, batch_size):
    """
    Run the model on the selected speech samples and return one predicted
    Bopomofo-ID sequence for each sample.

    Used for validation previews and displayed test predictions.
    """
    model.eval()
    predictions = []

    for start in range(0, len(indices), batch_size):
        idx = indices[start:start + batch_size]

        logits = model(X[idx])

        frame_predictions = logits.argmax(dim=-1)
        time_lengths = cnn_time_lengths[idx]

        for row, length in zip(frame_predictions, time_lengths):
            predictions.append(
                _ctc_output_to_bopomofo_ids(row[:int(length)].tolist(), blank_id)
            )

    return predictions


@torch.no_grad()
def evaluate_train_val(indices, model, X, bopomofo_id_lists, bopomofo_ids_padded,
                       bopomofo_lengths, cnn_time_lengths, blank_id, criterion,
                       device, batch_size, compute_loss=True):
    """
    Evaluate the training or validation data during fitting.

    Returns:
        CTC loss (optional), strict SER, strict syllable accuracy,
        strict whole-sentence accuracy
    """
    model.eval()

    predicted_id_lists = []
    correct_id_lists = []

    total_loss = 0.0
    total_samples_for_loss = 0

    for start in range(0, len(indices), batch_size):
        idx = indices[start:start + batch_size]

        logits = model(X[idx])

        if compute_loss:
            loss = ctc_loss(
                logits, idx, bopomofo_ids_padded, bopomofo_lengths,
                cnn_time_lengths, criterion, device
            )
            total_loss += float(loss) * len(idx)
            total_samples_for_loss += len(idx)

        frame_predictions = logits.argmax(dim=-1)
        time_lengths = cnn_time_lengths[idx]

        for row, length, sample_i in zip(
            frame_predictions, time_lengths, idx.detach().cpu().tolist()
        ):
            predicted_id_lists.append(
                _ctc_output_to_bopomofo_ids(row[:int(length)].tolist(), blank_id)
            )
            correct_id_lists.append(bopomofo_id_lists[sample_i])

    ser, syllable_accuracy, exact_accuracy = strict_metrics(
        predicted_id_lists, correct_id_lists
    )

    mean_loss = total_loss / total_samples_for_loss if compute_loss else None
    return mean_loss, ser, syllable_accuracy, exact_accuracy


@torch.no_grad()
def evaluate_test(indices, model, X, bopomofo_id_lists, bopomofo_ids_padded,
                  bopomofo_lengths, cnn_time_lengths, blank_id, criterion,
                  id_to_token, device, batch_size):
    """
    Final held-out test evaluation.

    Returns CTC loss plus both strict and Taiwan-normalized metrics.
    """
    model.eval()

    predicted_id_lists = []
    correct_id_lists = []

    total_loss = 0.0
    total_samples_for_loss = 0

    for start in range(0, len(indices), batch_size):
        idx = indices[start:start + batch_size]

        logits = model(X[idx])

        loss = ctc_loss(
            logits, idx, bopomofo_ids_padded, bopomofo_lengths,
            cnn_time_lengths, criterion, device
        )
        total_loss += float(loss) * len(idx)
        total_samples_for_loss += len(idx)

        frame_predictions = logits.argmax(dim=-1)
        time_lengths = cnn_time_lengths[idx]

        for row, length, sample_i in zip(
            frame_predictions, time_lengths, idx.detach().cpu().tolist()
        ):
            predicted_id_lists.append(
                _ctc_output_to_bopomofo_ids(row[:int(length)].tolist(), blank_id)
            )
            correct_id_lists.append(bopomofo_id_lists[sample_i])

    strict_ser, strict_syllable_acc, strict_exact_acc = strict_metrics(
        predicted_id_lists, correct_id_lists
    )
    tw_ser, tw_syllable_acc, tw_exact_acc = tw_metrics(
        predicted_id_lists, correct_id_lists, id_to_token
    )

    test_loss = total_loss / total_samples_for_loss

    return (
        test_loss,
        strict_ser,
        strict_syllable_acc,
        strict_exact_acc,
        tw_ser,
        tw_syllable_acc,
        tw_exact_acc
    )


# ============================================================
# Internal helpers
# ============================================================

def _ctc_output_to_bopomofo_ids(frame_ids, blank_id):
    """
    Convert frame-by-frame CTC output into one predicted Bopomofo-ID sequence.

    Example:
        [blank, 12, 12, blank, 7, 7, blank] -> [12, 7]

    Consecutive repeated IDs are collapsed and CTC blank IDs are removed.
    """
    ids = []
    previous = None

    for token_id in frame_ids:
        token_id = int(token_id)

        if token_id != blank_id and token_id != previous:
            ids.append(token_id)

        previous = token_id

    return ids
