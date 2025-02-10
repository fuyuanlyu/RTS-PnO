import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fourier import frequency_modes


__all__ = [
    'ChannelMixingRoutedAttention',
    'FourierEnhancedAttention',
    'AutoCorrelation',
    'ProbSparseAttention',
    'ScaledDotProductAttention',
    'MultiHeadLayer'
]


class ChannelMixingRoutedAttention(nn.Module):
    """Multi-head scaled dot-product attention with router mechanism for channel-mixing.
    Router mechanism (Crossformer, Zhang et al, 2023) ensures linear complexity (O(cL))
    which is essential for attention-based channel-mixing on datasets with more variates.
    """

    def __init__(self, n_patches, n_routers, d_model, n_heads, dropout):
        super().__init__()
        self.router = nn.Parameter(torch.randn(n_patches, n_routers, d_model))
        self.sender_attn = MultiHeadLayer(
            ScaledDotProductAttention(apply_casual_mask=False, dropout=dropout),
            d_model, n_heads, mix=True 
        )
        self.receiver_attn = MultiHeadLayer(
            ScaledDotProductAttention(apply_casual_mask=False, dropout=dropout),
            d_model, n_heads, mix=True 
        )

    def forward(self, x):
        """
        x   batch_size x n_vars x n_patches x d_model 
        """
        B, V, _, D = x.size()
        _, V_R, _ = self.router.size()

        router = self.router.unsqueeze(0).repeat(B, 1, 1, 1).view(-1, V_R, D)
        x = x.transpose(1, 2).contiguous().view(-1, V, D)

        buffer = self.sender_attn(router, x, x)
        context = self.receiver_attn(x, buffer, buffer)

        context = context.view(B, -1, V, D).transpose(1, 2).contiguous()
        return context


class FourierEnhancedAttention(nn.Module):
    """FEA-f used by (FEDformer, Zhou et al, 2022) to compute attention
    in the frequency domain after frequency mode filtering. 
    """

    def __init__(self, d_model, n_heads, q_seq_len, kv_seq_len, n_modes):
        super().__init__()
        self.d_model = d_model
        self.q_modes = frequency_modes(q_seq_len, n_modes)
        self.kv_modes = frequency_modes(kv_seq_len, n_modes)

        scale = 1 / (d_model * d_model)
        d_head = d_model // n_heads
        self.weights = nn.Parameter(
            scale * torch.rand(n_heads, d_head, d_head, len(self.q_modes), dtype=torch.cfloat)
        )

    def _filter_frequency_modes(self, x_fft, modes):
        B, H, D, _ = x_fft.size()

        x_out_fft = torch.zeros(B, H, D, len(modes), dtype=torch.cfloat, device=x_fft.device)
        for ix, mode in enumerate(modes):
            if ix >= x_out_fft.size(-1) or mode >= x_fft.size(-1):
                continue
            x_out_fft[:, :, :, ix] = x_fft[:, :, :, mode]

        return x_out_fft

    def forward(self, query, key, value):
        """
        query       batch_size x n_heads x q_seq_len x d_head
        key         batch_size x n_heads x kv_seq_len x d_head
        value       batch_size x n_heads x kv_seq_len x d_head
        """
        B, H, L, D = query.size()

        query = query.transpose(-2, -1).contiguous()
        key = key.transpose(-2, -1).contiguous()

        # Transform to frequency domain
        query_fft = torch.fft.rfft(query, dim=-1)
        key_fft = torch.fft.rfft(key, dim=-1)

        # Filter frequency modes
        query_fft = self._filter_frequency_modes(query_fft, self.q_modes)
        key_fft = self._filter_frequency_modes(key_fft, self.kv_modes)

        # Compute attention in the frequency domain
        scores_fft = torch.einsum('bhdq,bhdk->bhqk', query_fft, key_fft)
        attn_fft = scores_fft.tanh()

        context_fft = torch.einsum('bhqk,bhdk->bhdq', attn_fft, key_fft)
        context_fft = torch.einsum('bhiq,hijq->bhjq', context_fft, self.weights)

        # Fill in output context frequency modes
        context_out_fft = torch.zeros(B, H, D, L // 2 + 1, dtype=torch.cfloat, device=context_fft.device)
        for ix, mode in enumerate(self.q_modes):
            if ix >= context_fft.size(-1) or mode >= context_out_fft.size(-1):
                continue
            context_out_fft[:, :, :, mode] = context_fft[:, :, :, ix]

        # Transform to time domain
        context = torch.fft.irfft(context_out_fft / self.d_model / self.d_model, n=L, dim=-1)
        return context


class AutoCorrelation(nn.Module):
    """Auto-correlation mechanism (Autoformer, Wu et al, 2022)
    conducts sub-series level aggregation based on series periodicity (O(cLlogL)).
    """

    def __init__(self, factor):
        super().__init__()
        self.factor = factor

        self.correlation = None

    def forward(self, query, key, value):
        """
        query   batch_size x n_heads x q_seq_len x d_head
        key     batch_size x n_heads x kv_seq_len x d_head
        value   batch_size x n_heads x kv_seq_len x d_head
        """
        _, _, L_Q, _ = query.size()

        key, value = self._pad_or_truncate_key_value(key, value, L_Q)
        self.correlation = self._period_based_dependency(query, key)
        context = self._time_delay_aggregation(self.correlation, value)
        return context

    def _pad_or_truncate_key_value(self, key, value, L_Q):
        """Pad or truncate key and value to query's length."""
        _, _, L_KV, _ = key.size()

        if L_Q > L_KV:
            # Pad key and value to query's length with zeros
            key = F.pad(key, (0, 0, 0, L_Q - L_KV))
            value = F.pad(value, (0, 0, 0, L_Q - L_KV))
        else:
            # Truncate key and value to query's length
            key = key[:, :, :L_Q, :]
            value = value[:, :, :L_Q, :]

        return key, value

    def _period_based_dependency(self, query, key):
        """Efficient auto-correlation computation in the Fourier domain."""
        query_fft = torch.fft.rfft(query.transpose(-2, -1), dim=-1)
        key_fft = torch.fft.rfft(key.transpose(-2, -1), dim=-1)
        correlation_fft = query_fft * torch.conj(key_fft)
        correlation = torch.fft.irfft(correlation_fft, dim=-1)
        return correlation

    def _time_delay_aggregation(self, correlation, value):
        """Aggregates shifted value based on correlation to capture period-based dependency."""
        B, H, L_KV, D = value.size()

        # Average correlation across heads and channels
        mean_correlation = correlation.mean(dim=1).mean(dim=1)

        # Sample the top clog(L_KV) delays to compute aggregation weights
        num_delays = min(self.factor * math.ceil(math.log(L_KV)), L_KV)

        if self.training:
            # Average correlation across examples in the batch during training
            _, delay_index = torch.topk(mean_correlation.mean(dim=0), num_delays, dim=-1)
            delay_correlation = mean_correlation[:, delay_index]
            weights = torch.softmax(delay_correlation, dim=-1)

            # Aggregate lagged sequences with computed weights
            context = 0.
            for ix in range(num_delays):
                delayed_value = torch.roll(value, int(-delay_index[ix]), dims=-2)
                context += torch.einsum('b,bhvd->bhvd', weights[:, ix], delayed_value)
        else:
            # Example-wise correlation ranking during inference
            weights, delay_index = torch.topk(mean_correlation, num_delays, dim=-1)
            weights = torch.softmax(weights, dim=-1)

            # Aggregate lagged sequences with computed weights
            context = 0.
            value_expand = value.repeat(1, 1, 2, 1)
            init_index = torch.arange(L_KV, device=value.device).view(1, 1, -1, 1).expand(B, H, -1, D)
            for ix in range(num_delays):
                gather_index = init_index + delay_index[:, ix].view(-1, 1, 1, 1).expand(-1, H, L_KV, D)
                delayed_value = torch.gather(value_expand, dim=-2, index=gather_index)
                context += torch.einsum('b,bhvd->bhvd', weights[:, ix], delayed_value)

        return context


class ProbSparseAttention(nn.Module):
    """ProbSparse attention (Informer, Zhou et al, 2021)
    computes attention with queries sampled based on the sparsity measurement (O(cLlogL)).
    """

    def __init__(self, apply_causal_mask, factor):
        super().__init__()
        self.apply_causal_mask = apply_causal_mask
        self.factor = factor

        self.attention = None

    def forward(self, query, key, value):
        """
        query   batch_size x n_heads x q_seq_len x d_head
        key     batch_size x n_heads x kv_seq_len x d_head
        value   batch_size x n_heads x kv_seq_len x d_head
        """
        _, _, L_Q, _ = query.size()

        attention, query_index = self._probsparse_attention_weights(query, key)
        context = self._probsparse_context(attention, value, query_index, L_Q)
        self._register_attention_weights(attention, query_index, L_Q)
        return context

    def _probsparse_attention_weights(self, query, key):
        """Computes attention weights with sampled queries."""
        B, H, L_Q, D = query.size()
        _, _, L_KV, _ = key.size()

        # Sample clog(L_KV) keys to compute the sparsity measurement
        num_keys = min(self.factor * math.ceil(math.log(L_KV)), L_KV)
        key_index = torch.randint(L_KV, (L_Q, num_keys))
        key_sample = key.unsqueeze(-3).expand(-1, -1, L_Q, -1, -1)[
            :, :, torch.arange(L_Q).unsqueeze(1), key_index, :
        ]

        # Compute the sparsity measurement
        scores = torch.einsum('bhqd,bhqkd->bhqk', query, key_sample)
        sparsity = scores.max(dim=-1)[0] - scores.sum(dim=-1) / L_KV

        # Sample the top clog(L_Q) queries based on the sparsity measurement
        num_queries = min(self.factor * math.ceil(math.log(L_Q)), L_Q)
        _, query_index = sparsity.topk(num_queries, sorted=False)

        # Compute attention scores with sampled queries
        query_sample = query[
            torch.arange(B).view(-1, 1, 1), torch.arange(H).view(1, -1, 1), query_index, :
        ]
        scores = torch.einsum('bhqd,bhkd->bhqk', query_sample, key) / math.sqrt(D)

        if self.apply_causal_mask:
            causal_mask = _probsparse_causal_mask(query_index, L_Q)
            scores = scores.masked_fill(causal_mask, -math.inf)

        attention = torch.softmax(scores, dim=-1)
        return attention, query_index

    def _probsparse_context(self, attention, value, query_index, L_Q):
        """Computes context vectors with attention weights and value."""
        B, H, L_KV, _ = value.size()

        # Initialize context vectors with cumulative sum or mean of the values
        if self.apply_causal_mask:
            # Self-attention in decoder
            assert L_KV == L_Q
            context = value.cumsum(dim=-2)
        else:
            context = value.mean(dim=-2, keepdims=True).repeat(1, 1, L_Q, 1)

        # Fill in context vectors with sparse attention
        context[
            torch.arange(B).view(-1, 1, 1), torch.arange(H).view(1, -1, 1), query_index, :
        ] = torch.matmul(attention, value)
        return context

    def _register_attention_weights(self, attention, query_index, L_Q):
        B, H, _, L_KV = attention.size()

        attention_ = torch.ones(B, H, L_Q, L_KV, device=attention.device) / L_KV
        attention_[
            torch.arange(B).view(-1, 1, 1), torch.arange(H).view(1, -1, 1), query_index, :
        ] = attention
        self.attention = attention_


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention (Transformer, Vaswani et al, 2017)
    with optional residual attention (Realformer, He et al, 2020) (O(L^2)).
    """

    def __init__(self, apply_casual_mask, dropout, apply_residual=False):
        super().__init__()
        self.apply_casual_mask = apply_casual_mask
        self.apply_residual = apply_residual

        self.attention = None

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, prev_scores=None):
        """
        query           batch_size x n_heads x q_seq_len x d_head
        key             batch_size x n_heads x kv_seq_len x d_head
        value           batch_size x n_heads x kv_seq_len x d_head
        prev_scores     batch_size x n_heads x q_seq_len x kv_seq_len
        """
        _, _, L_Q, D = query.size()

        scores = torch.einsum('bhqd,bhkd->bhqk', query, key) / math.sqrt(D)

        if self.apply_residual and prev_scores is not None:
            scores += prev_scores

        if self.apply_casual_mask:
            causal_mask = _triangular_causal_mask(L_Q, query.device)
            scores = scores.masked_fill(causal_mask, -math.inf)

        self.attention = self.dropout(torch.softmax(scores, dim=-1))
        context = torch.einsum('bhqk,bhkd->bhqd', self.attention, value)
        return context, scores


class MultiHeadLayer(nn.Module):
    """Multi-head wrapper for attention and auto-correlation layers
    with optional residual from the previous layer (Realformer, He et al, 2020).
    """

    def __init__(self, layer, d_model, n_heads, mix=False):
        assert d_model % n_heads == 0
        super().__init__()
        self.layer = layer
        self.n_heads = n_heads
        self.mix = mix

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_context = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, prev_scores=None, return_scores=False):
        """
        query         batch_size x q_seq_len x d_model
        key           batch_size x kv_seq_len x d_model
        value         batch_size x kv_seq_len x d_model
        prev_scores   batch_size x n_heads x q_seq_len x kv_seq_len
        """
        B, L_Q, D = query.size()
        _, L_KV, _ = key.size()

        query = self.W_q(query).view(B, L_Q, self.n_heads, -1).transpose(1, 2).contiguous()
        key = self.W_k(key).view(B, L_KV, self.n_heads, -1).transpose(1, 2).contiguous()
        value = self.W_v(value).view(B, L_KV, self.n_heads, -1).transpose(1, 2).contiguous()

        if hasattr(self.layer, 'apply_residual'):
            context, scores = self.layer(query, key, value, prev_scores)
        else:
            context = self.layer(query, key, value)

        if not self.mix:
            context = context.transpose(1, 2).contiguous()

        context = context.view(B, L_Q, D)
        context = self.W_context(context)
        return (context, scores) if return_scores else context


def _triangular_causal_mask(seq_len, device):
    """Upper triangular causal mask."""
    ones = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    mask = torch.triu(ones, diagonal=1)
    return mask


def _probsparse_causal_mask(query_index, seq_len):
    """Upper triangular causal mask for sampled queries."""
    B, H, _ = query_index.size()

    mask = _triangular_causal_mask(seq_len, query_index.device)
    mask = mask.view(1, 1, seq_len, seq_len).expand(B, H, -1, -1)[
        torch.arange(B).view(-1, 1, 1), torch.arange(H).view(1, -1, 1), query_index, :
    ]
    return mask
