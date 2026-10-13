from .taiwan_bopomofo_equiv import taiwan_equivalent


def strict_metrics(predicted_id_lists, correct_id_lists):
    """
    Compare predicted and correct Bopomofo syllable-ID sequences exactly.

    Returns:
        SER                 = total insertions/deletions/substitutions / total correct syllables
        syllable_accuracy   = 1 - SER
        exact_accuracy      = fraction of speech samples whose whole Bopomofo sequence is correct
    """
    total_edits = 0
    total_correct_syllables = 0
    exact_matches = 0

    for predicted_ids, correct_ids in zip(predicted_id_lists, correct_id_lists):
        total_edits += _edit_distance(predicted_ids, correct_ids)
        total_correct_syllables += len(correct_ids)
        exact_matches += int(predicted_ids == correct_ids)

    ser = total_edits / total_correct_syllables
    syllable_accuracy = 1.0 - ser
    exact_accuracy = exact_matches / len(correct_id_lists)

    return ser, syllable_accuracy, exact_accuracy


def tw_metrics(predicted_id_lists, correct_id_lists, id_to_token):
    """
    Same evaluation as strict_metrics(), but selected Taiwan-pronunciation
    equivalents are also counted as matches. Tone must still match.
    """
    def tw_equal(predicted_id, correct_id):
        return taiwan_equivalent(
            id_to_token[int(predicted_id)],
            id_to_token[int(correct_id)]
        )

    total_edits = 0
    total_correct_syllables = 0
    exact_matches = 0

    for predicted_ids, correct_ids in zip(predicted_id_lists, correct_id_lists):
        total_edits += _edit_distance(predicted_ids, correct_ids, tw_equal)
        total_correct_syllables += len(correct_ids)

        exact_matches += int(
            len(predicted_ids) == len(correct_ids)
            and all(tw_equal(p, c) for p, c in zip(predicted_ids, correct_ids))
        )

    ser = total_edits / total_correct_syllables
    syllable_accuracy = 1.0 - ser
    exact_accuracy = exact_matches / len(correct_id_lists)

    return ser, syllable_accuracy, exact_accuracy


# ============================================================
# Internal helpers
# ============================================================

def _edit_distance(predicted_ids, correct_ids, equal_fn=None):
    """
    Minimum number of Bopomofo-syllable insertions, deletions and substitutions
    needed to change the prediction into the correct sequence.
    """
    if equal_fn is None:
        equal_fn = lambda a, b: a == b

    previous = list(range(len(correct_ids) + 1))

    for i, predicted in enumerate(predicted_ids, start=1):
        current = [i]

        for j, correct in enumerate(correct_ids, start=1):
            substitution = 0 if equal_fn(predicted, correct) else 1
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + substitution
            ))

        previous = current

    return previous[-1]
