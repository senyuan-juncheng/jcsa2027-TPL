def make_history_row(epoch, train_loss, train_metrics, val_metrics, seconds):
    """
    Put one epoch's train/validation results into one dictionary.
    The notebook appends these rows and converts them to a DataFrame after training.
    """
    train_ser, train_syllable_acc, train_exact_acc = train_metrics
    val_loss, val_ser, val_syllable_acc, val_exact_acc = val_metrics

    return {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_ser": train_ser,
        "train_syllable_acc": train_syllable_acc,
        "train_exact_acc": train_exact_acc,
        "val_loss": val_loss,
        "val_ser": val_ser,
        "val_syllable_acc": val_syllable_acc,
        "val_exact_acc": val_exact_acc,
        "seconds": seconds,
    }


def print_epoch(row):
    """Print the train/validation results for one epoch on one line."""
    print(
        f"epoch {row['epoch']:03d} | train-loss {row['train_loss']:.4f} | "
        f"train-syllable {row['train_syllable_acc']:.4f} | train-exact {row['train_exact_acc']:.4f} | "
        f"val-loss {row['val_loss']:.4f} | val-syllable {row['val_syllable_acc']:.4f} | "
        f"val-exact {row['val_exact_acc']:.4f} | {row['seconds']:.1f}s"
    )


def copy_model_state(model):
    """
    Copy the current model parameters to CPU.
    Used to remember the epoch with the lowest validation loss for early stopping.
    """
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
