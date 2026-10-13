import torch
from torch import nn


class TensorProjection(nn.Module):
    """
    Mode-wise tensor projection.

    Input:
        (batch, p1, ..., pd)

    Output:
        (batch, q1, ..., qd)

    Modes with pj == qj pass through unchanged.
    """

    def __init__(self, in_dims, out_dims, eps=1e-6):
        super().__init__()

        self.in_dims = tuple(in_dims)
        self.out_dims = tuple(out_dims)
        self.eps = eps
        self.weights = nn.ParameterDict()

        for mode, (p, q) in enumerate(zip(self.in_dims, self.out_dims)):
            if p == q:
                continue

            weight = nn.Parameter(torch.empty(p, q))
            nn.init.kaiming_uniform_(weight, a=5**0.5)
            self.weights[str(mode)] = weight

    def _polar_factor(self, weight):
        q = weight.shape[1]
        gram = weight.T @ weight + self.eps * torch.eye(q, device=weight.device)

        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        inv_sqrt = (eigenvectors * torch.rsqrt(eigenvalues).unsqueeze(0)) @ eigenvectors.T

        return weight @ inv_sqrt

    @staticmethod
    def _mode_product(x, matrix, axis):
        x = torch.tensordot(x, matrix, dims=([axis], [0]))
        return x.movedim(-1, axis)

    def forward(self, x):
        z = x

        for mode in range(len(self.in_dims)):
            key = str(mode)
            if key in self.weights:
                z = self._mode_product(z, self._polar_factor(self.weights[key]), mode + 1)

        return z
