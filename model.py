"""
model.py — Mimi-to-Wav2Vec2 Bridge Module
==========================================
Converts Mimi discrete token streams (B, T, 8) → Wav2Vec2-like features (B, 2T, 768).

Architecture:
  1. Multi-codebook token embeddings
  2. ConvTranspose1d temporal upsampling (×2)
     Mimi 12.5 Hz × 2 = 25 Hz = bridge output rate
  3. Causal Transformer with relative positional encoding
  4. Linear output projection to 768-dim  (matches facebook/wav2vec2-base-960h hidden size)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Positional Encodings
# ──────────────────────────────────────────────────────────────────────────────

class SinusoidalPE(nn.Module):
    """Standard sinusoidal positional encoding (non-learnable)."""

    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class RelativePositionBias(nn.Module):
    """
    T5-style relative position bias added to attention logits.
    Supports causal (unidirectional) masking.
    """

    def __init__(self, num_heads: int, max_distance: int = 128, causal: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.causal = causal
        self.embeddings = nn.Embedding(2 * max_distance + 1, num_heads)
        nn.init.normal_(self.embeddings.weight, std=0.02)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Returns (num_heads, T, T) bias tensor."""
        pos = torch.arange(seq_len, device=device)
        rel = pos.unsqueeze(0) - pos.unsqueeze(1)          # (T, T)
        rel = rel.clamp(-self.max_distance, self.max_distance) + self.max_distance
        bias = self.embeddings(rel)                         # (T, T, H)
        bias = bias.permute(2, 0, 1)                        # (H, T, T)
        if self.causal:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
            bias = bias.masked_fill(mask.unsqueeze(0), float("-inf"))
        return bias


# ──────────────────────────────────────────────────────────────────────────────
# Causal Multi-Head Attention with Optional Relative Bias
# ──────────────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
        use_relative_pe: bool = True,
        max_distance: int = 128,
    ):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = math.sqrt(self.head_dim)

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj  = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.use_relative_pe = use_relative_pe
        if use_relative_pe:
            self.rel_bias = RelativePositionBias(nhead, max_distance, causal=True)

        # KV-cache (used during inference)
        self._cache_k: Optional[torch.Tensor] = None
        self._cache_v: Optional[torch.Tensor] = None

    def reset_cache(self):
        self._cache_k = None
        self._cache_v = None

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.shape

        qkv = self.qkv_proj(x)  # (B, T, 3C)
        q, k, v = qkv.chunk(3, dim=-1)

        def split_heads(t):
            return t.view(B, -1, self.nhead, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        present_kv = (k, v) if use_cache else None
        S = k.size(2)  # full key/value length

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, T, S)

        if self.use_relative_pe:
            bias = self.rel_bias(S, x.device)  # (H, S, S) — causal mask inside
            # Only take last T rows for query side (during incremental decoding T < S)
            attn = attn + bias[:, -T:, :]

        else:
            # Standard causal mask
            causal_mask = torch.triu(
                torch.ones(T, S, device=x.device), diagonal=S - T + 1
            ).bool()
            attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)  # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_drop(self.out_proj(out))
        return out, present_kv


# ──────────────────────────────────────────────────────────────────────────────
# Transformer Layer & Stack
# ──────────────────────────────────────────────────────────────────────────────

class TransformerLayer(nn.Module):

    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float, use_relative_pe: bool):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, nhead, dropout, use_relative_pe)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        past_kv=None,
    ):
        # Pre-norm residual
        attn_out, present_kv = self.attn(self.norm1(x), use_cache=use_cache, past_kv=past_kv)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x, present_kv


class CausalTransformer(nn.Module):

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_ff: int,
        dropout: float,
        use_relative_pe: bool,
        max_seq_len: int,
    ):
        super().__init__()
        self.use_relative_pe = use_relative_pe
        if not use_relative_pe:
            self.pos_enc = SinusoidalPE(d_model, max_seq_len, dropout)

        self.layers = nn.ModuleList([
            TransformerLayer(d_model, nhead, dim_ff, dropout, use_relative_pe)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        past_kvs=None,
    ):
        if not self.use_relative_pe:
            x = self.pos_enc(x)

        present_kvs = []
        for i, layer in enumerate(self.layers):
            pkv = past_kvs[i] if past_kvs is not None else None
            x, pres = layer(x, use_cache=use_cache, past_kv=pkv)
            present_kvs.append(pres)

        x = self.norm(x)
        return x, present_kvs if use_cache else None


# ──────────────────────────────────────────────────────────────────────────────
# Multi-Codebook Embedding
# ──────────────────────────────────────────────────────────────────────────────

class MultiCodebookEmbedding(nn.Module):
    """
    8 separate embedding tables (one per codebook level).
    Fusion: element-wise sum → (B, T, embed_dim)
    """

    def __init__(
        self,
        num_codebooks: int = 8,
        vocab_size: int = 2048,
        embed_dim: int = 256,
        fusion: str = "sum",
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.embed_dim = embed_dim
        self.fusion = fusion

        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim)
            for _ in range(num_codebooks)
        ])

        if fusion == "concat":
            self.proj = nn.Linear(embed_dim * num_codebooks, embed_dim)

        # Level-wise learnable scale (weight each codebook differently)
        self.level_scale = nn.Parameter(torch.ones(num_codebooks))

        self._init_weights()

    def _init_weights(self):
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, T, num_codebooks)  integer indices
        returns: (B, T, embed_dim)
        """
        embeds = []
        for i, emb in enumerate(self.embeddings):
            e = emb(tokens[:, :, i])        # (B, T, embed_dim)
            e = e * self.level_scale[i]
            embeds.append(e)

        if self.fusion == "sum":
            out = torch.stack(embeds, dim=0).sum(dim=0)   # (B, T, embed_dim)
        elif self.fusion == "concat":
            out = torch.cat(embeds, dim=-1)               # (B, T, embed_dim * K)
            out = self.proj(out)                          # (B, T, embed_dim)
        else:
            raise ValueError(f"Unknown fusion: {self.fusion}")

        return out


# ──────────────────────────────────────────────────────────────────────────────
# Causal Temporal Upsampler (×2)
# ──────────────────────────────────────────────────────────────────────────────

class CausalUpsample(nn.Module):
    """
    ConvTranspose1d (×2) followed by a causal conv to prevent future leakage.
    Input:  (B, D, T)
    Output: (B, D, 2T)
    """

    def __init__(self, channels: int, upsample_factor: int = 4):
        super().__init__()
        self.factor = upsample_factor

        # Transposed conv for upsampling
        self.conv_t = nn.ConvTranspose1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=upsample_factor,
            stride=upsample_factor,
        )

        # Causal refinement conv (kernel=3, causal padding)
        self.causal_refine = nn.Conv1d(channels, channels, kernel_size=3, padding=0)
        self.causal_pad = 2  # (kernel - 1)

        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D, T)"""
        x = self.conv_t(x)                              # (B, D, 4T)
        # Causal padding: pad only on the left
        x_pad = F.pad(x, (self.causal_pad, 0))
        x = self.causal_refine(x_pad)                  # (B, D, 4T)
        x = self.act(self.norm(x))
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Main Bridge Module
# ──────────────────────────────────────────────────────────────────────────────

class MimiWav2Vec2Bridge(nn.Module):
    """
    Full Mimi → Wav2Vec2 bridge.

    Forward pass:
        tokens (B, T, 8) → features (B, 2T, 768)
        Mimi 12.5 Hz × upsample_factor 2 = 25 Hz = bridge output rate
    """

    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]

        self.embed_dim     = m["embed_dim"]
        self.d_model       = m["d_model"]
        self.output_dim    = m["output_dim"]
        self.upsample_factor = m["upsample_factor"]

        # 1. Multi-codebook embedding
        self.embedding = MultiCodebookEmbedding(
            num_codebooks=m["num_codebooks"],
            vocab_size=m["vocab_size"],
            embed_dim=m["embed_dim"],
            fusion=m["embed_fusion"],
        )

        # 2. Input projection (embed_dim → d_model if different)
        if m["embed_dim"] != m["d_model"]:
            self.input_proj = nn.Linear(m["embed_dim"], m["d_model"])
        else:
            self.input_proj = nn.Identity()

        # 3. Temporal upsampling
        self.upsample = CausalUpsample(m["d_model"], m["upsample_factor"])

        # 4. Causal Transformer
        use_rel = m["pos_encoding"] == "relative"
        self.transformer = CausalTransformer(
            d_model=m["d_model"],
            nhead=m["nhead"],
            num_layers=m["num_layers"],
            dim_ff=m["dim_feedforward"],
            dropout=m["dropout"],
            use_relative_pe=use_rel,
            max_seq_len=m["max_seq_len"],
        )

        # 5. Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(m["d_model"], m["d_model"]),
            nn.GELU(),
            nn.Linear(m["d_model"], m["output_dim"]),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        use_cache: bool = False,
        past_kvs=None,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        tokens:  (B, T, 8)   — integer Mimi indices
        returns: (B, 2T, output_dim) — Wav2Vec2-like features
        """
        # Embedding: (B, T, 8) → (B, T, embed_dim)
        x = self.embedding(tokens)

        # Optional projection to d_model
        x = self.input_proj(x)               # (B, T, d_model)

        # Upsample: (B, T, D) → (B, 4T, D) via conv
        x = x.transpose(1, 2)               # (B, D, T)
        x = self.upsample(x)                # (B, D, 4T)
        x = x.transpose(1, 2)               # (B, 4T, D)

        # Causal Transformer
        x, present_kvs = self.transformer(x, use_cache=use_cache, past_kvs=past_kvs)

        # Output projection
        x = self.output_proj(x)             # (B, 4T, output_dim)

        return x, present_kvs

    def get_param_count(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# Backward-compatibility alias — existing code importing MimiHuBERTBridge still works
MimiHuBERTBridge = MimiWav2Vec2Bridge


# ──────────────────────────────────────────────────────────────────────────────
# Discriminator (for adversarial loss)
# ──────────────────────────────────────────────────────────────────────────────

class FeatureDiscriminator(nn.Module):
    """
    Multi-scale 1D convolutional discriminator.
    Input: feature sequence (B, T, output_dim)  — default 768 for Wav2Vec2-base
    Output: scalar real/fake logits
    """

    def __init__(self, input_dim: int = 768, hidden: int = 512, num_layers: int = 4):
        super().__init__()
        layers = []
        in_ch = input_dim
        for i in range(num_layers):
            out_ch = hidden
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.1),
            ]
            in_ch = out_ch

        layers.append(nn.Conv1d(in_ch, 1, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D)"""
        x = x.transpose(1, 2)      # (B, D, T)
        return self.net(x)          # (B, 1, T')
