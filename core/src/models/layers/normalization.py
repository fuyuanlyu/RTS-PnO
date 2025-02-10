import torch
import torch.nn as nn


__all__ = ['RevIN']


class RevIN(nn.Module):
    """Reversible Instance Normalization (RevIN, Kim et al, 2022)
    normalizes the look-back window and denormalizes the predicted sequence
    with statistics of the look-back window instance-wise.
    """

    def __init__(self, num_features, affine, subtract_last, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.subtract_last = subtract_last
        self.eps = eps

        self.center = None
        self.scale = None

        if self.affine:
            self.weight = nn.Parameter(torch.ones(self.num_features))
            self.bias = nn.Parameter(torch.zeros(self.num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x, mode):
        """
        x   batch_size x seq_len x n_vars
        """
        assert mode in {'norm', 'denorm'}

        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)

        return x

    def _get_statistics(self, x):
        if self.subtract_last:
            self.center = x[:, [-1], :]
        else:
            self.center = torch.mean(x, dim=1, keepdim=True)
        self.scale = torch.sqrt(torch.var(x, dim=1, unbiased=False, keepdim=True) + self.eps)

    def _normalize(self, x):
        x = (x - self.center) / self.scale
        if self.affine:
            x = self.weight * x + self.bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.bias) / (self.weight + self.eps ** 2)
        x = x * self.scale + self.center
        return x


class SeasonalLayerNorm(nn.Module):
    """Specially designed layer normalization for the seasonal component.
    Used by (Autoformer, Wu et al, 2022) and (FEDformer, Zhou et al, 2022).
    """

    def __init__(self, num_features):
        super().__init__()
        self.norm = nn.LayerNorm(num_features)

    def forward(self, x):
        """
        x   batch_size x seq_len x d_model
        """
        x = self.norm(x)    
        bias = x.mean(dim=1).unsqueeze(1)
        return x - bias
