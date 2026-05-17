"""MAMMOTH mixture-of-experts module and supporting layers for publication use."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, einsum, nn
from torch.nn import Module

from einops import rearrange
from .components import ensure_batched, _kaiming_init, _kaiming_init_bias


def divisible_by(num: int, den: int) -> bool:
    """Return True if ``num`` is evenly divisible by ``den``."""
    return (num % den) == 0


def l2norm(t: Tensor, eps: float = 1e-8) -> Tensor:
    """L2-normalize the last dimension of a tensor."""
    return F.normalize(t, dim=-1, eps=eps)


class LayerNorm(nn.Module):
    """Layer normalization with learnable scale (gamma) and fixed zero bias (beta)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        """Normalize over the last dimension using gamma and beta."""
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class RMSNorm(Module):
    """Root-mean-square normalization: scale by dim^0.5 and learnable gamma."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """Apply RMS norm to the last dimension."""
        return l2norm(x) * self.scale * self.gamma


class ExpertWiseRMSNorm(nn.Module):
    """Per-expert RMS norm applied after concatenating heads."""

    def __init__(self, dim: int, num_heads: int, num_experts: int) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.scale = (dim * num_heads) ** 0.5
        self.gamma = nn.Parameter(torch.ones(num_experts, dim * num_heads))

    def forward(self, x: Tensor, eps: float = 1e-8) -> Tensor:
        """Apply expert-wise RMS norm after concatenating heads, then split back."""
        assert len(x.shape) == 4, "Expected 4 dimensional input"
        batch_size, num_experts, num_heads, dim = x.shape
        x_flat = x.reshape(batch_size, num_experts, -1)  # concat heads
        out = l2norm(x_flat, eps=eps) * self.scale * self.gamma.view(1, num_experts, -1)
        out = rearrange(out, "b e (h d) -> b e h d", h=num_heads, d=dim)
        return out


class FactorizedLinear(nn.Module):
    """
    Low-rank factorized linear layer: shared (or per-expert) projection to rank,
    then per-expert projection to out_dim. Supports optional weight sharing across experts.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int,
        num_experts: int,
        num_heads: int,
        share_weights: bool = True,
    ) -> None:
        super().__init__()
        self.share_weight = share_weights

        if share_weights:
            self.shared_weight = nn.Parameter(torch.randn(num_heads, in_dim, rank))
            self.shared_bias = nn.Parameter(torch.randn(num_heads, rank))
        else:
            self.shared_weight = nn.Parameter(
                torch.randn(num_experts, num_heads, in_dim, rank)
            )
            self.shared_bias = nn.Parameter(torch.randn(num_experts, num_heads, rank))

        self.expert_weights = nn.Parameter(
            torch.randn(num_experts, num_heads, rank, out_dim)
        )
        self.bias = nn.Parameter(torch.zeros(num_experts, num_heads, out_dim))

        _kaiming_init_bias(self.expert_weights, self.bias)
        _kaiming_init(self.shared_weight)
        _kaiming_init(self.expert_weights)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply factorized linear: x (..., in_dim) -> (..., out_dim).
        Input shape (S, E, H, in_dim); output (S, E, H, out_dim). Handles unbatched (S=1) input.
        """
        was_unbatched = len(x.shape) == 3
        if was_unbatched:
            x = x.unsqueeze(0)

        if self.share_weight:
            shared_weight = self.shared_weight.unsqueeze(0).unsqueeze(0)
            shared_component = torch.matmul(x.unsqueeze(-2), shared_weight).squeeze(-2)
            shared_component = shared_component + self.shared_bias.unsqueeze(
                0
            ).unsqueeze(0)
        else:
            shared_component = torch.matmul(
                x.unsqueeze(-2), self.shared_weight
            ).squeeze(-2)
            shared_component = shared_component + self.shared_bias.unsqueeze(
                0
            ).unsqueeze(0)

        factorized_component = torch.matmul(
            shared_component.unsqueeze(-2), self.expert_weights.unsqueeze(0)
        ).squeeze(-2)

        output = factorized_component + self.bias
        if was_unbatched:
            output = output.squeeze(0)
        return output


class MultiheadLinear(nn.Module):
    """
    Multi-head linear block: factorized linear, head-expert RMS norm, ReLU, dropout.
    Input per-head; in_features and out_features must be divisible by num_heads.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int,
        num_experts: int,
        lora_rank: int,
        dropout: float = 0.1,
        share_weights: bool = False,
    ) -> None:
        super().__init__()
        assert in_features % num_heads == 0, (
            "in_features must be divisible by num_heads"
        )
        assert out_features % num_heads == 0, (
            "out_features must be divisible by num_heads"
        )

        self.segment_dim = in_features // num_heads
        self.num_heads = num_heads
        self.out_features = out_features // num_heads
        self.num_experts = num_experts

        self.lora_layers = FactorizedLinear(
            self.segment_dim,
            self.out_features,
            lora_rank,
            num_experts,
            num_heads,
            share_weights,
        )
        self.activation = F.relu
        self.norm = ExpertWiseRMSNorm(
            self.out_features, self.num_heads, self.num_experts
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass. x shape: (S, E, H, D) with S=batch*slots, E=experts, H=heads, D=segment_dim.
        Returns tensor of shape (S, E, H, out_features//num_heads).
        """
        batch_size, num_experts, num_heads, segment_dim = x.shape
        assert (
            num_experts == self.num_experts
            and num_heads == self.num_heads
            and segment_dim == self.segment_dim
        ), "Input shape mismatch"

        out = self.lora_layers(x)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out


class Mammoth(nn.Module):
    """
    MAMMOTH mixture-of-experts layer: slot-based routing and multi-head factorized experts.
    Uses query projection, slot embeddings for routing, and optional LoRA-style factorized experts.
    """

    def __init__(
        self,
        input_dim: int,
        dim: int,
        *,
        num_experts: int = 30,
        num_slots: int = 10,
        num_heads: int = 16,
        dropout: float = 0.1,
        slot_dropout: float = 0.0,
        slot_dim: int = 256,
        lora_rank: int = 16,
        auto_rank: bool = True,
        keep_slots: bool = True,
        share_lora_weights: bool = True,
        use_layernorm: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: object,
    ) -> None:
        """
        Args:
            input_dim: Input feature dimension (must be divisible by num_heads).
            dim: Output feature dimension from expert heads.
            num_experts: Number of experts.
            num_slots: Slots per expert for routing.
            num_heads: Number of heads (input_dim and slot_dim must be divisible by this).
            dropout: Dropout probability in expert heads.
            slot_dropout: Dropout probability on slot logits in training.
            slot_dim: Query/slot projection dimension; if <= 0, uses dim.
            lora_rank: Rank of factorized expert linear layers.
            auto_rank: If True, compute lora_rank from parameter budget.
            keep_slots: If True, output all slot outputs concatenated; else combine with weights.
            share_lora_weights: Share low-rank weights across experts in factorized linear.
            use_layernorm: Use LayerNorm; if False, use RMSNorm.
            device: Optional device for expert_heads module.
        """
        super().__init__()
        assert num_experts >= 1, "expected >1 experts to use MAMMOTH"
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.num_heads = num_heads
        assert divisible_by(input_dim, num_heads), (
            "dimension must be divisible by number of heads"
        )

        assert dim != slot_dim, (
            "Output dimension must be different than slot dimension"
        )
        
        self.num_slots = num_slots
        self.keep_slots = keep_slots
        self.slot_dropout = slot_dropout

        if slot_dim <= 0:
            slot_dim = dim
        self.slot_dim = slot_dim

        if auto_rank:
            lora_rank = self._compute_auto_rank(
                input_dim, self.slot_dim, dim, num_experts
            )

        self.lora_rank = lora_rank

        norm_klass = LayerNorm if use_layernorm else RMSNorm

        self.head_dim = input_dim // num_heads
        self.head_dim_input = self.slot_dim // num_heads

        self.wq = nn.Linear(input_dim, self.slot_dim, bias=True)
        nn.init.xavier_uniform_(self.wq.weight)
        nn.init.zeros_(self.wq.bias)

        self.norm = norm_klass(self.slot_dim)
        self.slot_norm = norm_klass(self.head_dim_input)

        self.slot_embeds = nn.Parameter(
            torch.randn(num_experts, num_heads, num_slots, self.head_dim_input)
        )
        nn.init.orthogonal_(self.slot_embeds.to(torch.float32))
        nn.init.xavier_uniform_(self.slot_embeds)

        self.expert_heads = MultiheadLinear(
            self.slot_dim,
            dim,
            num_heads,
            num_experts,
            lora_rank,
            dropout,
            share_lora_weights,
        ).to(device)

    def _compute_auto_rank(
        self,
        input_dim: int,
        slot_dim: int,
        output_dim: int,
        num_experts: int,
    ) -> int:
        """Compute LoRA rank from parameter budget (input_dim, slot_dim, output_dim, num_experts).

        Args:
            input_dim: Input feature dimension.
            slot_dim: Slot projection dimension.
            output_dim: Output feature dimension.
            num_experts: Number of experts.

        Returns:
            Rank of the factorized linear layers.
        """
        num = input_dim * output_dim - input_dim * slot_dim
        denom = slot_dim + num_experts * output_dim
        rank = int((num / denom) + 0.5)
        print(
            f"Auto-computed LoRA rank: {rank} (from dimensions: input_dim={input_dim}, "
            f"slot_dim={slot_dim}, output_dim={output_dim}, num_experts={num_experts})"
        )
        return rank

    def forward(
        self, feats: Tensor, return_weights: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass. b=batch, n=seq, e=experts, s=slots, h=heads, d=dim.

        Args:
            feats: Input features, shape (batch, seq, input_dim) or (seq, input_dim).
            return_weights: If True, return the combine and dispatch weights.
        Returns:
            Output tensor of shape (batch, seq, num_experts * num_slots * head_dim) if keep_slots
            else (batch, seq, dim) after combining slots with combine_weights.
            If return_weights is True, return the dispatch weights.
        """
        feats, _ = ensure_batched(feats)
        b, n, d = feats.shape

        x = self.norm(self.wq(feats))
        x = rearrange(x, "b n (h d) -> b n h d", h=self.num_heads)

        logits = self.get_logits(x)
        combine_weights, dispatch_weights = self.get_weights(logits)

        slots = einsum("b n h d, b n e h s -> b e h s d", x, dispatch_weights)
        slots = rearrange(slots, "b e h s d -> (b s) e h d")

        out = self.expert_heads(slots)
        out = rearrange(
            out,
            "(b s) e h d -> b (e s) (h d)",
            e=self.num_experts,
            b=b,
            h=self.num_heads,
        )

        if not self.keep_slots:
            out = rearrange(
                out,
                "b (e s) (h d) -> b h (e s) d",
                e=self.num_experts,
                h=self.num_heads,
            )
            out = einsum("b h p d, b n h p -> b n h d", out, combine_weights)
            out = rearrange(out, "b n h d -> b n (h d)")

        if return_weights:
            return out, dispatch_weights
        return out

    def get_logits(self, x: Tensor) -> Tensor:
        """
        Compute routing logits between queries and slot embeddings.

        Args:
            x: Query tensor shape (b, n, h, d). slot_embeds: (e, h, s, d).

        Returns:
            Logits tensor shape (b, n, e, h, s).
        """
        slot_embeds = self.slot_norm(self.slot_embeds)
        logits = einsum("b n h d, e h s d -> b n e h s", x, slot_embeds)
        return logits

    def get_weights(self, logits: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Convert logits to combine and dispatch softmax weights. Optionally apply slot dropout in training.

        Args:
            logits: Shape (b, n, e, h, s).

        Returns:
            combine_weights: (b, n, h, e*s) for combining slot outputs. dispatch_weights: (b, n, e, h, s).
        """
        logits_f = logits.float()
        if self.slot_dropout > 0 and self.training:
            dropout_mask = (
                torch.rand(logits_f.shape, device=logits_f.device) > self.slot_dropout
            )
            dispatch_weights = F.softmax(
                logits_f.masked_fill(~dropout_mask, float("-inf")), dim=1
            ).to(logits.dtype)
            logits_reshaped = rearrange(logits_f, "b n e h s -> b n h (e s)")
            mask_reshaped = rearrange(dropout_mask, "b n e h s -> b n h (e s)")
            combine_weights = F.softmax(
                logits_reshaped.masked_fill(~mask_reshaped, float("-inf")), dim=-1
            )
        else:
            dispatch_weights = F.softmax(logits, dim=1)
            combine_weights = F.softmax(
                rearrange(logits, "b n e h s -> b n h (e s)"), dim=-1
            ).to(logits.dtype)

        return combine_weights, dispatch_weights
