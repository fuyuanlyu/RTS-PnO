import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.normalization import RevIN
from .layers.embedding import TimeSeriesChannelMixingEmbedding
from .layers.convolution import InceptionBlockV1


__all__ = ['TimesNet']


class TimesNet(nn.Module):
    """(TimesNet, Wu et al, 2023) applies CNNs on top of patched and unfolded 2D time series data
    to capture intra-and-inter period dependencies.
    """

    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len

        if configs.revin:
            self.revin = RevIN(configs.n_vars, configs.revin_affine, configs.revin_subtract_last)
        else:
            self.register_module('revin', None)

        # Module initialization order may affect forecasting performance (0.42 -> 0.39 on ETTh1 96)
        # Keep TimesBlock to be first initialized
        self.enc_layers = nn.ModuleList([
            TimesBlock(configs.topk, configs.d_model, configs.d_ff, configs.num_kernels, configs.activation)
            for _ in range(configs.e_layers)
        ])
        self.embed = TimeSeriesChannelMixingEmbedding(
            configs.n_vars, configs.d_model,
            configs.pos_enc, configs.temp_enc, configs.enc_freq,
            configs.dropout
        )
        self.norm = nn.LayerNorm(configs.d_model)

        self.enc_proj = nn.Linear(configs.seq_len, configs.seq_len + self.pred_len)
        self.proj = nn.Linear(configs.d_model, configs.n_vars)

    def forward(self, x_enc, x_stamp_enc, *args, **kwargs):
        """
        x_enc           batch_size x seq_len x n_vars
        x_stamp_enc     batch_size x seq_len x n_temp_feats
        """
        # RevIN
        if self.revin is not None:
            x_enc = self.revin(x_enc, mode='norm')

        # Embedding & Temporal projection
        x_embed = self.embed(x_enc, x_stamp_enc)
        x_embed = self.enc_proj(x_embed.transpose(-2, -1)).transpose(-2, -1).contiguous()

        # Encoder
        enc_out = x_embed
        for enc_layer in self.enc_layers:
            enc_out = self.norm(enc_layer(enc_out))

        # Output projection
        enc_out = self.proj(enc_out)

        # RevIN
        if self.revin is not None:
            enc_out = self.revin(enc_out, mode='denorm')

        return enc_out[:, -self.pred_len:, :] 


class TimesBlock(nn.Module):

    def __init__(self, topk, d_model, d_ff, num_kernels, activation):
        assert activation in {'relu', 'gelu'}
        super().__init__()
        self.topk = topk

        self.conv = nn.Sequential(
            InceptionBlockV1(d_model, d_ff, num_kernels),
            (nn.ReLU if activation == 'relu' else nn.GELU)(),
            InceptionBlockV1(d_ff, d_model, num_kernels)
        )

    def _topk_periodicities(self, x, topk):
        x_fft = torch.fft.rfft(x, dim=1)
        amplitude = torch.abs(x_fft).mean(0).mean(1)
        amplitude[0] = 0

        _, freq_top = torch.topk(amplitude, topk)
        freq_top = freq_top.detach().cpu().numpy()

        period_list = x.size(1) // freq_top
        period_weight = F.softmax(torch.abs(x_fft).mean(2)[:, freq_top], dim=-1)
        return period_list, period_weight

    def _multiresolution_patching(self, x, period_list):
        _, L, _ = x.size()
        x = x.transpose(-2, -1).contiguous()

        x_patch_list = []
        for period in period_list:
            x_pad = x
            pad_len = L % period
            if pad_len != 0:
                pad_len = period - pad_len
                x_pad = F.pad(x, (0, pad_len), mode='constant')

            x_patch = x_pad.unfold(dimension=-1, size=period, step=period)
            x_patch_list.append(x_patch)

        return x_patch_list

    def forward(self, x):
        """
        x   batch_size x seq_len x d_model
        """
        B, L, D = x.size()

        period_list, period_weight = self._topk_periodicities(x, self.topk)
        x_patch_list = self._multiresolution_patching(x, period_list)           

        x_out = []
        for x_patch in x_patch_list:
            x_patch = self.conv(x_patch)
            x_patch = x_patch.permute(0, 2, 3, 1).contiguous().view(B, -1, D)
            x_out.append(x_patch[:, :L, :])

        x_out = torch.stack(x_out, dim=1)
        x_ = torch.einsum('bkvl,bk->bvl', x_out, period_weight)

        x = x_ + x
        return x
