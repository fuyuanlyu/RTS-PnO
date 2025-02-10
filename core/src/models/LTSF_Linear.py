import torch
import torch.nn as nn

from .layers.decomposition import SeasonalTrendDecomposition


__all__ = ['DLinear', 'NLinear', 'Linear']


class DLinear(nn.Module):
    """(LTSF-Linear, Zeng et al, 2023) with seasonal-trend decomposition."""

    def __init__(self, configs):
        super().__init__()
        self.decomp = SeasonalTrendDecomposition(configs.decomp_ksize)
        self.proj_seasonal = Linear(configs)
        self.proj_trend = Linear(configs)

    def forward(self, x_enc, *args, **kwargs):
        """
        x   batch_size x seq_len x n_vars
        """
        seasonal_init, trend_init = self.decomp(x_enc)
        seasonal_out = self.proj_seasonal(seasonal_init)
        trend_out = self.proj_trend(trend_init)
        return seasonal_out + trend_out


class NLinear(nn.Module):
    """(LTSF-Linear, Zeng et al, 2023) with subtract-last normalization."""

    def __init__(self, configs):
        super().__init__()
        self.proj = Linear(configs)

    def forward(self, x_enc, *args, **kwargs):
        """
        x   batch_size x seq_len x n_vars
        """
        seq_last = x_enc[:, [-1], :]
        x = x_enc - seq_last
        x = self.proj(x)
        x = x + seq_last
        return x


class Linear(nn.Module):
    """(LTSF-Linear, Zeng et al, 2023) models dependencies with a linear layer."""

    def __init__(self, configs):
        super().__init__()
        self.n_vars = configs.n_vars
        self.pred_len = configs.pred_len
        self.shared_proj = configs.shared_proj

        if self.shared_proj:
            self.proj = nn.Linear(configs.seq_len, self.pred_len)
        else:
            self.proj = nn.ModuleList()
            for _ in range(self.n_vars):
                self.proj.append(nn.Linear(configs.seq_len, self.pred_len))

    def forward(self, x_enc, *args, **kwargs):
        """
        x   batch_size x seq_len x n_vars
        """
        if self.shared_proj:
            x = self.proj(x_enc.transpose(-2, -1)).transpose(-2, -1).contiguous()
        else:
            x = torch.stack([self.proj[ix](x_enc[:, :, ix]) for ix in range(self.n_vars)], dim=2)

        return x
