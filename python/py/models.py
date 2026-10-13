import torch.nn as nn

from .tensor_projection import TensorProjection


class Baseline(nn.Module):
    """
    Baseline ASR model.

    log-mel
      -> CNN block 1: F/2, T/2
      -> CNN block 2: F/4, T/4
      -> CNN block 3 + MaxPool(4,1): F/16, T/4
      -> flatten channel x frequency
      -> BiGRU
      -> Bopomofo logits
    """

    def __init__(self, n_mels, vocab_size, c1, c2, c3, gru_hidden, gru_layers, dropout):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d((2, 2))
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d((2, 2))
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d((4, 1))
        )

        gru_input_size = c3 * (n_mels // 16)

        self.bigru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0
        )

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(2 * gru_hidden, vocab_size)

    def forward(self, x):
        x = self.conv1(x)                  # (B, c1, F/2, T/2)
        x = self.conv2(x)                  # (B, c2, F/4, T/4)
        x = self.conv3(x)                  # (B, c3, F/16, T/4)

        x = x.permute(0, 3, 1, 2)          # (B, T/4, c3, F/16)
        x = x.flatten(start_dim=2)          # (B, T/4, c3*F/16)

        x, _ = self.bigru(x)
        x = self.dropout(x)
        return self.head(x)


class TPL(nn.Module):
    """
    TPL ASR model.

    The CNN blocks are identical to the Baseline up to the third
    convolution. The final dimension-reduction step differs:

        Baseline: MaxPool(4,1)
                  (c3, F/4, T/4) -> (c3, F/16, T/4)

        TPL     : TensorProjection reduces both channel and frequency
                  dimensions by a factor of 2:
                  (c3, F/4, T/4) -> (c3/2, F/8, T/4)

    Both models therefore feed the same flattened feature size into
    the BiGRU:

        c3 * F/16 = (c3/2) * F/8.
    """

    def __init__(self, n_mels, max_time, vocab_size, c1, c2, c3,
                 gru_hidden, gru_layers, dropout):
        super().__init__()

        if c3 % 2 != 0:
            raise ValueError("c3 must be even because TPL reduces the channel dimension by 2.")

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d((2, 2))
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d((2, 2))
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.GELU()
        )

        time_after_cnn = max_time // 4
        self.tpl = TensorProjection(
            in_dims=(c3, n_mels // 4, time_after_cnn),
            out_dims=(c3 // 2, n_mels // 8, time_after_cnn)
        )

        gru_input_size = (c3 // 2) * (n_mels // 8)

        self.bigru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0
        )

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(2 * gru_hidden, vocab_size)

    def forward(self, x):
        x = self.conv1(x)                  # (B, c1, F/2, T/2)
        x = self.conv2(x)                  # (B, c2, F/4, T/4)
        x = self.conv3(x)                  # (B, c3, F/4, T/4)
        x = self.tpl(x)                    # (B, c3/2, F/8, T/4)

        x = x.permute(0, 3, 1, 2)          # (B, T/4, c3/2, F/8)
        x = x.flatten(start_dim=2)          # (B, T/4, (c3/2)*(F/8))

        x, _ = self.bigru(x)
        x = self.dropout(x)
        return self.head(x)
