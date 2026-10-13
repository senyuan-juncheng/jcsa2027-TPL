import numpy as np
import torch


def encode_bopomofo_ids(meta, token_to_id, blank_id):
    """
    Convert the correct Bopomofo syllables of every speech sample to vocab IDs.

    For each sample:
        Chinese sentence
            -> Bopomofo syllables created in Steps 2-3
            -> Bopomofo vocab IDs used by CTC

    The ID sequences have different lengths, so a padded 2D array is also created
    for convenient storage on the GPU.
    """
    bopomofo_id_lists = [
        [token_to_id[token] for token in text.split()]
        for text in meta["bopomofo_tokens"].fillna("").astype(str)
    ]

    bopomofo_lengths = np.array([len(ids) for ids in bopomofo_id_lists], dtype=np.int64)

    max_length = int(bopomofo_lengths.max())
    bopomofo_ids_padded = np.full((len(bopomofo_id_lists), max_length), blank_id, dtype=np.int64)

    for i, ids in enumerate(bopomofo_id_lists):
        bopomofo_ids_padded[i, :len(ids)] = ids

    return bopomofo_id_lists, bopomofo_lengths, bopomofo_ids_padded


def remove_bopomofo_padding(batch_idx, bopomofo_ids_padded, bopomofo_lengths, device):
    """
    Remove the padding before CTCLoss.

    bopomofo_ids_padded stores all correct ID sequences as one rectangular matrix.
    bopomofo_lengths tells us how many IDs are actually correct for each sample.

    The returned 1D tensor contains only the real Bopomofo IDs from the selected batch.
    Padding values are not passed to CTCLoss as correct labels.
    """
    lengths = bopomofo_lengths[batch_idx]
    padded = bopomofo_ids_padded[batch_idx]

    positions = torch.arange(padded.shape[1], device=device).unsqueeze(0)
    valid = positions < lengths.unsqueeze(1)

    return padded[valid], lengths
