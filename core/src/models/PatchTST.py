import torch
import torch.nn as nn

from .layers.normalization import RevIN
from .layers.embedding import TimeSeriesPatchEmbedding
from .layers.decomposition import SeasonalTrendDecomposition
from .layers.attention import ScaledDotProductAttention, MultiHeadLayer


__all__ = ['PatchTST']


class PatchTST(nn.Module):
    """(PatchTST, Nie et al, 2023) with vanilla transformer encoder
    applied with patch-based embedding and channel-independent modeling (O(L^2/S^2)).
    """

    def __init__(self, configs):
        super().__init__()

        if configs.decomp:
            self.decomp = SeasonalTrendDecomposition(configs.decomp_ksize)
            self.ptst_trend = PatchTSTBackbone(configs)
            self.ptst_seasonal = PatchTSTBackbone(configs)
        else:
            self.register_module('decomp', None)
            self.ptst = PatchTSTBackbone(configs)

    def forward(self, x_enc, *args, **kwargs):
        """
        x_enc   batch_size x seq_len x n_vars
        """
        if x_enc.dim() == 2: # Single Variable prediction
            x_enc = x_enc.unsqueeze(2)
        assert x_enc.dim() == 3
        if self.decomp is not None:
            seasonal_init, trend_init = self.decomp(x_enc)
            seasonal = self.ptst_seasonal(seasonal_init)
            trend = self.ptst_trend(trend_init)
            enc_out = seasonal + trend
        else:
            enc_out = self.ptst(x_enc)

        if enc_out.dim() == 3: # Squeeze output for Single Variable prediction
            if enc_out.size(dim=2) == 1:
                enc_out = enc_out.squeeze()
        return enc_out


class PatchTSTBackbone(nn.Module):
    """PatchTST backbone."""

    def __init__(self, configs):
        super().__init__()

        if configs.revin:
            self.revin = RevIN(configs.n_vars, configs.revin_affine, configs.revin_subtract_last)
        else:
            self.register_module('revin', None)

        self.embed = TimeSeriesPatchEmbedding(
            configs.patch_len, configs.patch_stride, configs.patch_padding,
            configs.n_vars, configs.d_model, configs.pos_enc
        )
        self.dropout_embed = nn.Dropout(configs.dropout)

        self.encoder = PatchTSTEncoder(
            configs.e_layers,
            configs.d_model, configs.n_heads,
            configs.d_ff, configs.activation,
            configs.dropout
        )

        n_patches = int((configs.seq_len - configs.patch_len) / configs.patch_stride) + 1
        n_patches += configs.patch_padding == 'end'
        in_features = n_patches * configs.d_model
        self.proj = PatchTSTPredictionHead(
            in_features, configs.n_vars, configs.pred_len,
            configs.shared_proj
        )

    def forward(self, x):
        """
        x   batch_size x seq_len x n_vars
        """
        # RevIN
        if self.revin is not None:
            x = self.revin(x, mode='norm')

        # Encoder
        x_embed = self.embed(x)
        x = self.encoder(x_embed)

        # Output projection
        enc_out = self.proj(x)

        # RevIN
        if self.revin is not None:
            enc_out = self.revin(enc_out, mode='denorm')

        return enc_out


class PatchTSTEncoder(nn.Module):
    """PatchTST encoder."""

    def __init__(self, e_layers, d_model, n_heads, d_ff, activation, dropout):
        super().__init__()
        self.enc_layers = nn.ModuleList([
            PatchTSTEncoderLayer(d_model, n_heads, d_ff, activation, dropout)
            for _ in range(e_layers)
        ])

    def forward(self, x):
        """
        x   batch_size x n_vars x n_patches x d_model
        """
        _, _, L, D = x.size()

        x = x.view(-1, L, D)
        scores = None
        for enc_layer in self.enc_layers:
            x, scores = enc_layer(x, scores)

        return x


class PatchTSTPredictionHead(nn.Module):
    """PatchTST prediction head."""

    def __init__(self, in_features, n_vars, pred_len, shared_proj):
        super().__init__()
        self.n_vars = n_vars
        self.shared_proj = shared_proj

        if self.shared_proj:
            self.proj = nn.Linear(in_features, pred_len)
        else:
            self.proj = nn.ModuleList()
            for _ in range(self.n_vars):
                self.proj.append(nn.Linear(in_features, pred_len))

    def forward(self, x):
        """
        x   (batch_size * n_vars) x n_patches x d_model
        """
        _, L, D = x.size()

        x = x.view(-1, self.n_vars, L * D)
        if self.shared_proj:
            x = self.proj(x).transpose(-2, -1).contiguous()
        else:
            x = torch.stack([self.proj[ix](x[:, ix, :]) for ix in range(self.n_vars)], dim=2)

        return x


class PatchTSTEncoderLayer(nn.Module):
    """PatchTST encoder layer."""

    def __init__(self, d_model, n_heads, d_ff, activation, dropout):
        assert activation in {'relu', 'gelu'}
        super().__init__()
        self.self_attn = MultiHeadLayer(
            ScaledDotProductAttention(apply_casual_mask=False, dropout=dropout, apply_residual=True),
            d_model, n_heads
        )
        self.dropout_self_attn = nn.Dropout(dropout)
        self.norm_self_attn = nn.BatchNorm1d(d_model)

        self.mlp_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            (nn.ReLU if activation == 'relu' else nn.GELU)(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout_mlp_ffn = nn.Dropout(dropout)
        self.norm_mlp_ffn = nn.BatchNorm1d(d_model)

    def forward(self, x, prev_scores):
        """
        x               (batch_size * n_vars) x n_patches x d_model
        prev_scores     (batch_size * n_vars) x n_patches x n_patches 
        """
        # Self-attention + BatchNorm
        x_, scores = self.self_attn(x, x, x, prev_scores, return_scores=True)
        x = self.norm_self_attn(
            (x + self.dropout_self_attn(x_)).transpose(-2, -1)
        ).transpose(-2, -1).contiguous()

        # Position-wise feed-forward + BatchNorm
        x = self.norm_mlp_ffn(
            (x + self.dropout_mlp_ffn(self.mlp_ffn(x))).transpose(-2, -1)
        ).transpose(-2, -1).contiguous()
        return x, scores
