import torch
import torch.nn as nn
import torch.nn.functional as F

from .decomposition import SeasonalTrendDecomposition


__all__ = ['MultiScaleIsometricConvolution', 'InceptionBlockV1']


class MultiScaleIsometricConvolution(nn.Module):
    """MIC layer used by (MICN, Wang et al, 2023) to consecutively capture
    local and global dependencies.    
    """

    def __init__(self, d_model, d_ff, decomp_ksizes, conv_ksizes, isoconv_ksizes, activation, dropout):
        super().__init__()
        assert len(decomp_ksizes) == len(conv_ksizes) == len(isoconv_ksizes)
        assert activation in {'relu', 'gelu'}
        self.num_scales = len(decomp_ksizes)

        # Seasonal-trend decomposition
        self.decomps = nn.ModuleList([
            SeasonalTrendDecomposition(decomp_ksize)
            for decomp_ksize in decomp_ksizes
        ])

        # Up-and-down sampling convolutions for local-global dependency modeling
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, conv_ksize, stride=conv_ksize, padding=conv_ksize//2)
            for conv_ksize in conv_ksizes
        ])
        self.isoconvs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, isoconv_ksize, stride=1, padding=0)
            for isoconv_ksize in isoconv_ksizes
        ])
        self.trans_convs = nn.ModuleList([
            nn.ConvTranspose1d(d_model, d_model, conv_ksize, stride=conv_ksize, padding=0)
            for conv_ksize in conv_ksizes
        ])
        self.act_conv = nn.Tanh()
        self.norm_conv = nn.LayerNorm(d_model)
        self.dropout_conv = nn.Dropout(dropout)

        # Convolutional aggregation of multiple scales
        self.agg_conv = nn.Conv2d(d_model, d_model, (self.num_scales, 1))

        # Position-wise feed-forward
        self.mlp_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            (nn.ReLU if activation == 'relu' else nn.GELU)(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm_mlp_ffn = nn.LayerNorm(d_model)

    def _local_global_dependency(self, x, ix):
        _, L, _ = x.size()

        # Downsampling convolution to model local dependencies
        x_ = x.transpose(-2, -1).contiguous()
        x_down = self.dropout_conv(self.act_conv(self.convs[ix](x_)))

        # Isometric convolution to model global dependencies
        x_down_pad = F.pad(x_down, (x_down.size(2) - 1, 0), mode='constant')
        x_down = self.norm_conv(
            (x_down + self.dropout_conv(self.act_conv(self.isoconvs[ix](x_down_pad)))).transpose(-2, -1)
        ).transpose(-2, -1)

        # Upsampling convolution to recover the original sequence length
        x_ = self.dropout_conv(self.act_conv(self.trans_convs[ix](x_down)))
        x_ = x_[:, :, :L].transpose(-2, -1).contiguous()
        x = self.norm_conv(x + x_)

        return x

    def forward(self, x):
        """
        x   batch_size x seq_len x d_model
        """
        # Multi-scale local-global dependency modeling
        x_list = []
        for ix in range(self.num_scales):
            x_, _ = self.decomps[ix](x)
            x_ = self._local_global_dependency(x_, ix)
            x_list.append(x_)

        # Convolutional aggregation of multi-scale results
        x = torch.stack(x_list, dim=1)
        x = self.agg_conv(x.permute(0, 3, 1, 2)).squeeze(-2).transpose(-2, -1).contiguous()

        # Position-wise feed forward
        x = self.norm_mlp_ffn(x + self.mlp_ffn(x))
        return x


class InceptionBlockV1(nn.Module):
    """Inception block used by (TimesNet, Wu et al, 2023) to jointly capture
    intra-and-inter period dependencies on reshaped time series data.
    """

    def __init__(self, in_channels, out_channels, num_kernels):
        super().__init__()

        self.conv_layers = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=2*ix+1, stride=1, padding=ix)
            for ix in range(num_kernels)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """
        x   batch_size x seq_len x n_vars
        """
        x_list = []
        for conv_layer in self.conv_layers:
            x_list.append(conv_layer(x))

        x = torch.stack(x_list, dim=-1).mean(-1)
        return x
