from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import torch
from torch import nn

if TYPE_CHECKING:
    from jaxtyping import Float


class ScalarEmbedder(nn.Module):
    """Sinusoidal embedding for conditioning scalars (VIX, returns, etc)."""

    def __init__(self, scalar_dim: int, embed_dim: int, frequency_dim: int = 64, max_period: int = 10000) -> None:
        super().__init__()
        self.frequency_dim = frequency_dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim * scalar_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def sinusoidal_embedding(self, x: Float[torch.Tensor, "B S"]) -> Float[torch.Tensor, "B S*F"]:
        """Create sinusoidal embeddings for each scalar dimension."""
        half = self.frequency_dim // 2
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(half, dtype=torch.float32, device=x.device) / half)
        # x: (B, S), freqs: (half,) -> args: (B, S, half)
        args = x[:, :, None] * freqs[None, None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, S, frequency_dim)
        return embedding.flatten(1)  # (B, S * frequency_dim)

    def forward(self, scalars: Float[torch.Tensor, "B S"]) -> Float[torch.Tensor, "B D"]:
        emb = self.sinusoidal_embedding(scalars)
        return self.mlp(emb)


class FourierFeatureEmbedder(nn.Module):
    """Maps low-dim coordinates to high-dim Fourier features."""

    def __init__(self, in_dim: int, embed_dim: int, scale: float = 10.0) -> None:
        super().__init__()
        self.register_buffer("w", torch.randn(in_dim, embed_dim // 2) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = torch.einsum("...i,ij->...j", x, self.w)  # x @ self.w
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int = 2,
        act: Literal["gelu", "silu"] = "silu",
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        act_fn = nn.SiLU() if act == "silu" else nn.GELU()

        for i in range(n_layers):
            if i == 0:
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(act_fn)
            elif i == n_layers - 1:
                layers.append(nn.Linear(hidden_dim, out_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(act_fn)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GeometricGatedFFN(nn.Module):
    def __init__(self, embed_dim: int, space_dim: int, n_experts: int = 4) -> None:
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(space_dim, 64),
            nn.SiLU(),
            nn.Linear(64, n_experts),
            nn.Softmax(dim=-1),
        )

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Linear(embed_dim * 4, embed_dim),
                )
                for _ in range(n_experts)
            ]
        )

    def forward(self, x: Float[torch.Tensor, "B N D"], pos: Float[torch.Tensor, "B N 2"]) -> Float[torch.Tensor, "B N D"]:
        weights = self.gate_net(pos).unsqueeze(-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
        return (weights * expert_outputs).sum(dim=2)


class LinearHeterogeneousCrossAttention(nn.Module):
    """Linear attention with heterogeneous branch processing."""

    def __init__(self, embed_dim: int, n_heads: int = 4, n_branches: int = 3) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.query_proj = nn.Linear(embed_dim, embed_dim)

        self.key_projs = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(n_branches)])
        self.val_projs = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(n_branches)])

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self, x_trunk: Float[torch.Tensor, "B N D"], z_branches: list[Float[torch.Tensor, "B M D"]]
    ) -> Float[torch.Tensor, "B N D"]:
        B, N, D = x_trunk.shape

        q = self.query_proj(x_trunk).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        q = q.softmax(dim=-1)

        out = torch.zeros_like(q)

        for i, z in enumerate(z_branches):
            k = self.key_projs[i](z).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.val_projs[i](z).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.softmax(dim=-1)

            context = k.transpose(-2, -1) @ v
            k_cumsum = k.sum(dim=-2, keepdim=True)
            d_inv = 1.0 / (q * k_cumsum).sum(dim=-1, keepdim=True).clamp(min=1e-6)

            branch_out = (q @ context) * d_inv
            out = out + branch_out

        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)


class CrossAttentionBlock(nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        embed_dim: int,
        cond_dim: int,
        n_heads: int = 4,
        dropout: float = 0.0,
        space_dim: int = 2,
        n_branches: int = 3,
        n_experts: int = 4,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim, elementwise_affine=True)
        self.ln2_branches = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(n_branches)])
        self.ln3 = nn.LayerNorm(embed_dim)
        self.ln4 = nn.LayerNorm(embed_dim, elementwise_affine=True)
        self.ln5 = nn.LayerNorm(embed_dim)

        self.cross_attn = LinearHeterogeneousCrossAttention(embed_dim, n_heads, n_branches)
        self.self_attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)

        self.moe_mlp1 = GeometricGatedFFN(embed_dim, space_dim, n_experts)
        self.moe_mlp2 = GeometricGatedFFN(embed_dim, space_dim, n_experts)

        self.cond_proj = nn.Linear(cond_dim, 6 * embed_dim)
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        branches: list[torch.Tensor],
        cond: torch.Tensor,
        x_pos: torch.Tensor,
    ) -> torch.Tensor:
        (shift1, scale1, gate1, shift2, scale2, gate2) = self.cond_proj(cond).chunk(6, dim=-1)

        h_branches = [self.ln2_branches[i](b) for i, b in enumerate(branches)]
        h = self.ln1(x) * (1 + scale1[:, None, :]) + shift1[:, None, :]
        x = x + gate1[:, None, :] * self.cross_attn(h, h_branches)

        x_moe1 = self.moe_mlp1(x, x_pos)
        x = x + self.ln3(x_moe1)

        h = self.ln4(x) * (1 + scale2[:, None, :]) + shift2[:, None, :]
        h, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + gate2[:, None, :] * h

        x_moe2 = self.moe_mlp2(x, x_pos)
        return x + self.ln5(x_moe2)


class DiffusionGNOT(nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        space_dim: int = 2,
        value_dim: int = 1,
        context_value_dim: int = 3,
        scalar_dim: int = 6,
        embed_dim: int = 256,
        n_layers: int = 6,
        n_heads: int = 4,
        n_experts: int = 4,
        mlp_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.coord_embed = FourierFeatureEmbedder(space_dim, embed_dim, scale=20.0)
        self.scalar_embed = ScalarEmbedder(scalar_dim, embed_dim)
        self.input_proj = nn.Linear(value_dim + 1, embed_dim)

        self.branch_projs = nn.ModuleList(
            [MLP(in_dim=1, hidden_dim=embed_dim, out_dim=embed_dim, n_layers=mlp_layers) for _ in range(context_value_dim)]
        )
        self.ctx_coord_embed = FourierFeatureEmbedder(space_dim, embed_dim, scale=20.0)

        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    embed_dim=embed_dim,
                    cond_dim=embed_dim,
                    n_heads=n_heads,
                    dropout=dropout,
                    space_dim=space_dim,
                    n_branches=context_value_dim,
                    n_experts=n_experts,
                )
                for _ in range(n_layers)
            ]
        )

        self.input_norm = nn.LayerNorm(embed_dim)
        self.out_norm = nn.LayerNorm(embed_dim)
        self.out_mlp = MLP(embed_dim, embed_dim, 1, n_layers=mlp_layers)

    def forward(  # noqa: PLR0913
        self,
        query_coords: Float[torch.Tensor, "B N 2"],
        noisy_values: Float[torch.Tensor, "B N 1"],
        noise_scale: Float[torch.Tensor, "B N 1"],
        scalars: Float[torch.Tensor, "B S"],
        context_coords: Float[torch.Tensor, "B M 2"],
        context_values: Float[torch.Tensor, "B M 3"],
    ) -> Float[torch.Tensor, "B N 1"]:
        pos_emb = self.coord_embed(query_coords)
        val_noise = torch.cat([noisy_values, noise_scale], dim=-1)
        val_emb = self.input_proj(val_noise)
        x = val_emb + pos_emb
        x = self.input_norm(x)

        branch_vals_list = context_values.chunk(context_values.shape[-1], dim=-1)
        ctx_pos_emb = self.ctx_coord_embed(context_coords)

        encoded_branches = []
        for i, branch_val in enumerate(branch_vals_list):
            b_emb = self.branch_projs[i](branch_val) + ctx_pos_emb
            encoded_branches.append(b_emb)

        cond = self.scalar_embed(scalars)

        for block in self.blocks:
            x = block(x, encoded_branches, cond, query_coords)

        x = self.out_norm(x)
        return self.out_mlp(x)
