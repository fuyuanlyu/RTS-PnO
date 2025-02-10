import math

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['TimeSeriesPatchEmbedding', 'TimeSeriesEmbedding']


class TimeSeriesPatchEmbedding(nn.Module):
    """Patch-based time series embedding.
        - pos_enc: Type of positional encoding.
            none: No positional encoding
            sincos: Fixed sinusoidal positional encoding
            learned: Learnable positional encoding
            learned_per_var: Learnable positional encoding for each variate
    """

    def __init__(self, patch_len, stride, padding, n_vars, d_model, pos_enc):
        assert padding in {'none', 'start', 'end'}
        assert pos_enc in {'none', 'sincos', 'learned', 'learned_per_var'}
        super().__init__()

        self.embed_patch = PatchEmbedding(patch_len, stride, padding, d_model)
        if pos_enc != 'none':
            n_vars_tmp = n_vars if pos_enc == 'learned_per_var' else None
            self.enc_pos = PositionalEncoding(d_model, pos_enc, n_vars=n_vars_tmp)
        else:
            self.register_module('enc_pos', None)


    def forward(self, x):
        """
        x   batch_size x seq_len x n_vars
        """
        x = self.embed_patch(x)
        if self.enc_pos is not None:
            x += self.enc_pos(x)

        return x


class PatchEmbedding(nn.Module):
    """Segments time series into patches and embeds patches."""

    def __init__(self, patch_len, stride, padding, d_model):
        assert padding in {'none', 'start', 'end'}
        super().__init__()

        self.patch_len = patch_len
        self.stride = stride
        self.padding = padding

        self.embed = nn.Linear(patch_len, d_model)

    def forward(self, x):
        """
        x   batch_size x seq_len x n_vars 
        """
        _, L, _ = x.size()

        x = x.transpose(-2, -1).contiguous()

        # Patch segmentation
        if self.padding == 'start':
            pad_start = (L - self.patch_len) % self.stride
            if pad_start != 0:
                pad_start = self.stride - pad_start
                x = F.pad(x, (pad_start, 0), mode='replicate')
        elif self.padding == 'end':
            x = F.pad(x, (0, self.stride), mode='replicate')

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        # Patch embedding
        x = self.embed(x)
        return x


class TimeSeriesChannelMixingEmbedding(nn.Module):
    """Channel-mixing-based time series embedding.
        - pos_enc: Type of positional encoding.
            none: No positional encoding
            sincos: Fixed sinusoidal positional encoding
            learned: Learnable positional encoding
            learned_per_var: Learnable positional encoding for each variate
        - temp_enc: Type of temporal encoding.
            none: No temporal encoding
            sincos: Fixed sinusoidal timestamp encoding
            learned: Learnable timestamp encoding
            learned_proj: Learnable time feature encoding based on encoded timestamp
    """

    def __init__(self, n_vars, d_model, pos_enc, temp_enc, enc_freq, dropout):
        assert pos_enc in {'none', 'sincos', 'learned', 'learned_per_var'}
        assert temp_enc in {'none', 'sincos', 'learned', 'learned_proj'}
        assert (temp_enc == 'learned_proj' and isinstance(enc_freq, str)) or \
               (temp_enc != 'learned_proj' and enc_freq is None)
        super().__init__()

        self.embed_val = ChannelMixingEmbedding(n_vars, d_model)
        if pos_enc != 'none':
            n_vars_tmp = n_vars if pos_enc == 'learned_per_var' else None
            self.enc_pos = PositionalEncoding(d_model, pos_enc, n_vars=n_vars_tmp)
        else:
            self.register_module('enc_pos', None)
        if temp_enc != 'none':
            self.enc_temp = TemporalEncoding(d_model, temp_enc, enc_freq)
        else:
            self.register_module('enc_temp', None)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_stamp):
        """
        x           batch_size x seq_len x n_vars
        x_stamp     batch_size x seq_len x n_temp_feats
        """
        x = self.embed_val(x)
        if self.enc_pos is not None:
            x += self.enc_pos(x)
        if self.enc_temp is not None:
            x += self.enc_temp(x_stamp)

        return self.dropout(x)


class ChannelMixingEmbedding(nn.Module):
    """1D convolutional channel-mixing value embedding."""

    def __init__(self, n_vars, d_model):
        super().__init__()
        self.embed = nn.Conv1d(
            n_vars, d_model, kernel_size=3,
            padding='same', padding_mode='circular', bias=False
        )
        nn.init.kaiming_normal_(self.embed.weight)

    def forward(self, x):
        """
        x   batch_size x seq_len x n_vars
        """
        x = self.embed(x.transpose(1, 2)).transpose(1, 2).contiguous()
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding.
        - pos_enc: Type of positional encoding.
            sincos: Fixed sinusoidal positional encoding
            learned: Learnable positional encoding
            learned_per_var: Learnable positional encoding for each variate
    """

    def __init__(self, d_model, pos_enc, max_len=1500, n_vars=None):
        assert pos_enc in {'sincos', 'learned', 'learned_per_var'}
        assert (pos_enc != 'learned_per_var' and n_vars is None) or \
               (pos_enc == 'learned_per_var' and n_vars is not None)
        super().__init__()
        self.pe = _positional_encoding(max_len, d_model, pos_enc, n_vars=n_vars)

    def forward(self, x):
        """
        x   batch_size x seq_len x d_model
            batch_size x n_vars x n_patches x d_model
        """
        L = x.size(-2)
        return self.pe[..., :L, :]


class TemporalEncoding(nn.Module):
    """Encodes timestamp information.
        - temp_enc: Type of temporal encoding.
            sincos: Fixed sinusoidal timestamp encoding
            learned: Learnable timestamp encoding
            learned_proj: Learnable time feature encoding based on encoded timestamp
    """

    def __init__(self, d_model, temp_enc, enc_freq):
        assert temp_enc in {'sincos', 'learned_embed', 'learned_proj'}
        assert (temp_enc == 'learned_proj' and isinstance(enc_freq, str)) or \
               (temp_enc != 'learned_proj' and enc_freq is None)
        super().__init__()

        if temp_enc in {'sincos', 'learned'}:
            self.enc_temp = TimeStampEncoding(d_model, temp_enc)
        elif temp_enc == 'learned_proj':
            self.enc_temp = TimeFeatureEncoding(d_model, enc_freq)

    def forward(self, x):
        """
        x   batch_size x seq_len x n_temp_feats
        """
        return self.enc_temp(x)


class TimeStampEncoding(nn.Module):
    """Directly encodes timestamp as temporal encoding.
        - temp_enc: Type of timestamp encoding.
            sincos: Fixed sinusoidal timestamp encoding
            learned: Learnable timestamp encoding
    """

    def __init__(self, d_model, temp_enc):
        assert temp_enc in {'sincos', 'learned'}
        super().__init__()

        # The order here must align with that in
        # `data_provider.LTF_provider.LongTermForecastingDataset`
        if temp_enc == 'sincos':
            self.enc_temp = nn.ModuleList([
                nn.Embedding.from_pretrained(_positional_encoding(13, d_model, 'sincos')),   # Month
                nn.Embedding.from_pretrained(_positional_encoding(32, d_model, 'sincos')),   # Day
                nn.Embedding.from_pretrained(_positional_encoding(7, d_model, 'sincos')),    # Weekday
                nn.Embedding.from_pretrained(_positional_encoding(24, d_model, 'sincos')),   # Hour
                nn.Embedding.from_pretrained(_positional_encoding(4, d_model, 'sincos'))     # Minute
            ])
        elif temp_enc == 'learned':
            self.enc_temp = nn.ModuleList([
                nn.Embedding(13, d_model),      # Month
                nn.Embedding(32, d_model),      # Day
                nn.Embedding(7, d_model),       # Weekday
                nn.Embedding(24, d_model),      # Hour
                nn.Embedding(4, d_model)        # Minute
            ])

    def forward(self, x):
        """
        x   batch_size x seq_len x n_temp_feats
        """
        _, _, D = x.size()

        x_ = 0.
        x = x.long()
        for ix in range(D):
            x_ += self.enc_temp[ix](x[:, :, ix])

        return x_


class TimeFeatureEncoding(nn.Module):
    """Learnable time feature embedding based on encoded timestamp."""

    def __init__(self, d_model, enc_freq):
        super().__init__()
        from gluonts.time_feature import time_features_from_frequency_str
        num_features = len(time_features_from_frequency_str(enc_freq))
        self.enc_temp = nn.Linear(num_features, d_model, bias=False)

    def forward(self, x):
        """
        x   batch_size x seq_len x n_temp_feats
        """
        return self.enc_temp(x)


def _positional_encoding(max_len, d_model, pos_enc, n_vars=None):
    assert pos_enc in {'sincos', 'learned', 'learned_per_var'}
    assert (pos_enc != 'learned_per_var' and n_vars is None) or \
           (pos_enc == 'learned_per_var' and n_vars is not None)

    if pos_enc == 'sincos':
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        requires_grad = False
    elif pos_enc == 'learned':
        pe = torch.empty(max_len, d_model)
        nn.init.uniform_(pe, a=-0.02, b=0.02)
        requires_grad = True
    elif pos_enc == 'learned_per_var':
        pe = torch.empty(n_vars, max_len, d_model)
        nn.init.normal_(pe, mean=0., std=1.)
        requires_grad = True

    return nn.Parameter(pe, requires_grad)
