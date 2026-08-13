"""The sequence gate and P0L1's optimiser applied to it.

The GRU reads the raw window, the head sees that state next to the two expert
forecasts, and a sigmoid gives one scalar per origin.  No hidden MLP, no
attention, no normalisation layer.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from . import spec as S


class SequenceGate(nn.Module):
    def __init__(self, horizon: int, channels: int = 3, hidden: int = None):
        super().__init__()
        hidden = S.ENCODER["hidden_size"] if hidden is None else hidden
        self.gru = nn.GRU(channels, hidden, num_layers=S.ENCODER["num_layers"],
                          batch_first=True, bidirectional=False)
        self.head = nn.Linear(hidden + 2 * horizon, 1)

    def forward(self, sequence: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        _, state = self.gru(sequence)
        return torch.sigmoid(self.head(torch.cat([state[-1], context], dim=1))).squeeze(1)


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def fit_sequence_gate(sequence: np.ndarray, context: np.ndarray, block, device,
                      seed: int = None):
    """P0L1's Adam, learning rate, epoch count and loss, on the sequence gate.

    The only departure is that the epoch's gradient is accumulated over fixed
    row chunks instead of being formed in one allocation.  The parameter update
    is the same one a single full-batch step would produce; the chunking exists
    because a GRU over 96 steps cannot hold every fold in memory at once.
    """
    seed = S.TRAINING["seed"] if seed is None else seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SequenceGate(horizon=context.shape[1] // 2).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=S.TRAINING["lr"])

    seq = torch.tensor(sequence, dtype=torch.float32, device=device)
    ctx = torch.tensor(context, dtype=torch.float32, device=device)
    sse = torch.tensor(block["sse_a"].to_numpy(np.float64), dtype=torch.float32, device=device)
    num = torch.tensor(block["num"].to_numpy(np.float64), dtype=torch.float32, device=device)
    den = torch.tensor(block["den"].to_numpy(np.float64), dtype=torch.float32, device=device)
    n_obs = torch.tensor(block["n"].to_numpy(np.float64), dtype=torch.float32, device=device)

    rows = len(seq)
    chunk = S.TRAINING["chunk_rows"]
    for _ in range(S.TRAINING["epochs"]):
        optimiser.zero_grad(set_to_none=True)
        for start in range(0, rows, chunk):
            stop = min(start + chunk, rows)
            g = model(seq[start:stop], ctx[start:stop])
            loss = ((sse[start:stop] - 2 * g * num[start:stop]
                     + g ** 2 * den[start:stop]) / n_obs[start:stop]).sum() / rows
            loss.backward()
        optimiser.step()
    model.eval()
    return model


@torch.no_grad()
def gate_weights(model, sequence: np.ndarray, context: np.ndarray, device) -> np.ndarray:
    chunk = S.TRAINING["chunk_rows"]
    out = np.empty(len(sequence), dtype=np.float64)
    for start in range(0, len(sequence), chunk):
        stop = min(start + chunk, len(sequence))
        seq = torch.tensor(sequence[start:stop], dtype=torch.float32, device=device)
        ctx = torch.tensor(context[start:stop], dtype=torch.float32, device=device)
        out[start:stop] = model(seq, ctx).detach().cpu().numpy().astype(np.float64)
    return out
