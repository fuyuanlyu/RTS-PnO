import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['MultiScaleHybridDecomposition', 'SeasonalTrendDecomposition']


class MultiScaleHybirdDecomposition(nn.Module):
    """Decomposes time series into seasonal and trend-cyclical components
    with composed kernels (MICN, Wang et al, 2023).
    """

    def __init__(self, kernel_sizes):
        super().__init__()
        self.decomps = nn.ModuleList([
            SeasonalTrendDecomposition(kernel_size) for kernel_size in kernel_sizes
        ])

    def forward(self, x):
        """
        x   batch_size x seq_len x n_vars
        """
        seasonals = []
        trends = []
        for decomp in self.decomps:
            seasonal, trend = decomp(x)
            seasonals.append(seasonal)
            trends.append(trend)

        seasonal = sum(seasonals) / len(seasonals)
        trend = sum(trends) / len(trends)
        return seasonal, trend


class MixtureOfExpertsDecomposition(nn.Module):
    """Decomposes time series into seasonal and trend-cyclical components
    with multiple kernels weighted in a data-dependent fashion (FEDformer, Zhou et al, 2022).
    """

    def __init__(self, kernel_sizes):
        super().__init__()
        self.decomps = nn.ModuleList([
            SeasonalTrendDecomposition(kernel_size) for kernel_size in kernel_sizes
        ])
        self.linear = nn.Linear(1, len(kernel_sizes))

    def forward(self, x):
        """
        x   batch_size x seq_len x n_vars
        """
        trends = []
        for decomp in self.decomps:
            _, trend = decomp(x)
            trends.append(trend)

        trends = torch.stack(trends, dim=-1)
        weights = F.softmax(self.linear(x.unsqueeze(-1)), dim=-1)
        trend = torch.sum(trends * weights, dim=-1)

        seasonal = x - trend
        return seasonal, trend


class SeasonalTrendDecomposition(nn.Module):
    """Decomposes time series into seasonal and trend-cyclical components."""

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.moving_avg = nn.AvgPool1d(kernel_size, stride=1)

    def forward(self, x):
        """
        x   batch_size x seq_len x n_vars
        """
        front = x[:, [0], :].repeat(1, self.kernel_size - 1 - (self.kernel_size - 1) // 2, 1)
        end = x[:, [-1], :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_pad = torch.cat([front, x, end], dim=1)

        trend = self.moving_avg(x_pad.transpose(1, 2)).transpose(1, 2)
        seasonal = x - trend
        return seasonal, trend
