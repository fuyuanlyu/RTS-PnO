import numpy as np
import torch
import torch.nn as nn


__all__ = ['FourierEnhancedBlock', 'frequency_modes']


class FourierEnhancedBlock(nn.Module):
    """FEB-f block used by (FEDformer, Zhou et al, 2022) to learn feature transformation
    in the frequency domain after frequency mode filtering.
    """

    def __init__(self, d_model, n_heads, seq_len, n_modes):
        super().__init__()
        self.modes = frequency_modes(seq_len, n_modes)

        scale = 1 / (d_model * d_model)
        d_head = d_model // n_heads
        self.weights = nn.Parameter(
            scale * torch.rand(n_heads, d_head, d_head, len(self.modes), dtype=torch.cfloat)
        )

    def forward(self, x, *args):
        """
        x       batch_size x n_heads x seq_len x d_head
        """
        B, H, L, D = x.size()

        x = x.transpose(-2, -1).contiguous()
        x_fft = torch.fft.rfft(x, dim=-1)

        x_out_fft = torch.zeros(B, H, D, L // 2 + 1, dtype=torch.cfloat, device=x.device)
        for ix, mode in enumerate(self.modes):
            if ix >= x_out_fft.size(-1) or mode >= x_fft.size(-1):
                continue
            x_out_fft[:, :, :, ix] = torch.einsum('bhi,hij->bhj', x_fft[:, :, :, mode], self.weights[:, :, :, ix])

        x = torch.fft.irfft(x_out_fft, n=L, dim=-1)
        return x


def frequency_modes(seq_len, n_modes):
    """Randomly select modes in the frequency domain."""
    n_modes = min(n_modes, seq_len // 2)

    all_modes = list(range(0, seq_len // 2))
    np.random.shuffle(all_modes)
    modes = all_modes[:n_modes]

    modes.sort()
    return modes
