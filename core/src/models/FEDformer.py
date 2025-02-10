import torch
import torch.nn as nn

from .layers.embedding import TimeSeriesChannelMixingEmbedding
from .layers.normalization import SeasonalLayerNorm
from .layers.decomposition import MixtureOfExpertsDecomposition
from .layers.fourier import FourierEnhancedBlock
from .layers.attention import FourierEnhancedAttention, MultiHeadLayer


__all__ = ['FEDformer']


class FEDformer(nn.Module):
    """(FEDformer, Wu et al, 2022) with frequency-enhanced modules
    for frequency domain aggregation with O(LlogL) complexity.
    """

    def __init__(self, configs):
        super().__init__()
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        self.decomp = MixtureOfExpertsDecomposition(configs.decomp_ksizes)

        self.enc_embed = TimeSeriesChannelMixingEmbedding(
            configs.n_vars, configs.d_model,
            configs.pos_enc, configs.temp_enc, configs.enc_freq,
            configs.dropout
        )
        self.dec_embed = TimeSeriesChannelMixingEmbedding(
            configs.n_vars, configs.d_model,
            configs.pos_enc, configs.temp_enc, configs.enc_freq,
            configs.dropout
        )

        self.encoder = FEDformerEncoder(
            configs.e_layers,
            configs.seq_len,
            configs.d_model, configs.n_heads, configs.n_modes,
            configs.d_ff, configs.decomp_ksizes, configs.activation,
            configs.dropout
        )
        self.decoder = FEDformerDecoder(
            configs.d_layers,
            configs.seq_len, configs.seq_len // 2 + configs.pred_len,
            configs.d_model, configs.n_heads, configs.n_modes,
            configs.d_ff, configs.n_vars, configs.decomp_ksizes, configs.activation,
            configs.dropout
        )

        self.proj_seasonal = nn.Linear(configs.d_model, configs.n_vars)

    def forward(self, x_enc, x_stamp_enc, x_dec, x_stamp_dec):
        """
        x_enc           batch_size x enc_seq_len x n_vars
        x_stamp_enc     batch_size x enc_seq_len x n_temp_feats
        x_dec           batch_size x dec_seq_len x n_vars
        x_stamp_dec     batch_size x dec_seq_len x n_temp_feats
        """
        # Decompose encoder input as start tokens for decoder
        seasonal_init, trend_init = self.decomp(x_enc)
        mean = torch.mean(x_enc, dim=1, keepdim=True).expand(-1, self.pred_len, -1)
        zeros = torch.zeros_like(x_dec[:, -self.pred_len:, :])

        # Initialize decoder input
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)

        # Encoder
        x_enc_embed = self.enc_embed(x_enc, x_stamp_enc)
        memory = self.encoder(x_enc_embed)

        # Decoder
        x_dec_embed = self.dec_embed(seasonal_init, x_stamp_dec)
        seasonal, trend = self.decoder(x_dec_embed, trend_init, memory)

        # Output projection
        seasonal = self.proj_seasonal(seasonal[:, -self.pred_len:, :])
        dec_out = seasonal + trend[:, -self.pred_len:, :]
        return dec_out


class FEDformerEncoder(nn.Module):
    """FEDformer encoder."""

    def __init__(self, e_layers, seq_len, d_model, n_heads, n_modes, d_ff, kernel_sizes, activation, dropout):
        super().__init__()
        self.enc_layers = nn.ModuleList([
            FEDformerEncoderLayer(seq_len, d_model, n_heads, n_modes, d_ff, kernel_sizes, activation, dropout)
            for _ in range(e_layers)
        ])
        self.norm = SeasonalLayerNorm(d_model)

    def forward(self, x):
        """
        x   batch_size x enc_seq_len x d_model
        """
        for enc_layer in self.enc_layers:
            x = enc_layer(x)

        x = self.norm(x)
        return x


class FEDformerEncoderLayer(nn.Module):
    """FEDformer encoder layer with progressive decomposition architecture."""

    def __init__(self, seq_len, d_model, n_heads, n_modes, d_ff, kernel_sizes, activation, dropout):
        assert activation in {'relu', 'gelu'}
        super().__init__()
        self.self_attn = MultiHeadLayer(
            FourierEnhancedBlock(d_model, n_heads, seq_len, n_modes),
            d_model, n_heads, mix=True
        )
        self.dropout_self_attn = nn.Dropout(dropout)
        self.decomp_self_attn = MixtureOfExpertsDecomposition(kernel_sizes)

        self.conv_ffn = nn.Sequential(
            nn.Conv1d(d_model, d_ff, kernel_size=1, bias=False),
            (nn.ReLU if activation == 'relu' else nn.GELU)(),
            nn.Dropout(dropout),
            nn.Conv1d(d_ff, d_model, kernel_size=1, bias=False)
        )
        self.dropout_conv_ffn = nn.Dropout(dropout)
        self.decomp_conv_ffn = MixtureOfExpertsDecomposition(kernel_sizes)

    def forward(self, x):
        """
        x   batch_size x enc_seq_len x d_model
        """
        # Self-attention + Progressive decomposition
        x, _ = self.decomp_self_attn(
            x + self.dropout_self_attn(self.self_attn(x, x, x))
        )

        # Position-wise feed-forward + Progressive decomposition
        x, _ = self.decomp_conv_ffn(
            x + self.dropout_conv_ffn(self.conv_ffn(x.transpose(-2, -1)).transpose(-2, -1))
        )
        return x


class FEDformerDecoder(nn.Module):
    """FEDformer decoder."""

    def __init__(self, d_layers, enc_seq_len, dec_seq_len, d_model, n_heads, n_modes, d_ff, n_vars, kernel_size, activation, dropout):
        super().__init__()
        self.dec_layers = nn.ModuleList([
            FEDformerDecoderLayer(enc_seq_len, dec_seq_len, d_model, n_heads, n_modes, d_ff, n_vars, kernel_size, activation, dropout)
            for _ in range(d_layers)
        ])
        self.norm = SeasonalLayerNorm(d_model)

    def forward(self, x, trend, memory):
        """
        x           batch_size x dec_seq_len x d_model
        trend       batch_size x dec_seq_len x d_model
        memory      batch_size x enc_seq_len x d_model
        """
        for dec_layer in self.dec_layers:
            x, residual_trend = dec_layer(x, memory)
            trend = trend + residual_trend

        x = self.norm(x)
        return x, trend


class FEDformerDecoderLayer(nn.Module):
    """FEDformer decoder layer with progressive decomposition architecture."""

    def __init__(self, enc_seq_len, dec_seq_len, d_model, n_heads, n_modes, d_ff, n_vars, kernel_sizes, activation, dropout):
        assert activation in {'relu', 'gelu'}
        super().__init__()
        self.self_attn = MultiHeadLayer(
            FourierEnhancedBlock(d_model, n_heads, dec_seq_len, n_modes),
            d_model, n_heads, mix=True
        )
        self.dropout_self_attn = nn.Dropout(dropout)
        self.decomp_self_attn = MixtureOfExpertsDecomposition(kernel_sizes)

        self.cross_attn = MultiHeadLayer(
            FourierEnhancedAttention(d_model, n_heads, dec_seq_len, enc_seq_len, n_modes),
            d_model, n_heads, mix=True
        )
        self.dropout_cross_attn = nn.Dropout(dropout)
        self.decomp_cross_attn = MixtureOfExpertsDecomposition(kernel_sizes)

        self.conv_ffn = nn.Sequential(
            nn.Conv1d(d_model, d_ff, kernel_size=1, bias=False),
            (nn.ReLU if activation == 'relu' else nn.GELU)(),
            nn.Dropout(dropout),
            nn.Conv1d(d_ff, d_model, kernel_size=1, bias=False)
        )
        self.dropout_conv_ffn = nn.Dropout(dropout)
        self.decomp_conv_ffn = MixtureOfExpertsDecomposition(kernel_sizes)

        self.proj_trend = nn.Conv1d(
            d_model, n_vars, kernel_size=3,
            padding=1, padding_mode='circular',
            bias=False
        )

    def forward(self, x, memory):
        """
        x           batch_size x dec_seq_len x d_model
        memory      batch_size x enc_seq_len x d_model
        """
        # Self-attention + Progressive decomposition
        x, trend1 = self.decomp_self_attn(
            x + self.dropout_self_attn(self.self_attn(x, x, x))
        )

        # Cross-attention + Progressive decomposition
        x, trend2 = self.decomp_cross_attn(
            x + self.dropout_cross_attn(self.cross_attn(x, memory, memory))
        )

        # Position-wise feed-forward + Progressive decomposition
        x, trend3 = self.decomp_conv_ffn(
            x + self.dropout_conv_ffn(self.conv_ffn(x.transpose(-2, -1)).transpose(-2, -1))
        )

        # Synthesize trends and project
        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.proj_trend(residual_trend.transpose(-2, -1)).transpose(-2, -1)
        return x, residual_trend
