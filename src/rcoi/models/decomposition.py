from __future__ import annotations

import torch
from torch import nn


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("moving_average_kernel must be odd")
        self.kernel_size = kernel_size
        self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = (self.kernel_size - 1) // 2
        front = x[:, :1].repeat(1, pad)
        end = x[:, -1:].repeat(1, pad)
        padded = torch.cat([front, x, end], dim=1).unsqueeze(1)
        return self.pool(padded).squeeze(1)
