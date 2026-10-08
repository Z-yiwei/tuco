import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb


class RotaryPosEmb(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 256, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        theta = 1.0 / (base ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
        self.register_buffer("theta", theta, persistent=False)
        self._build_rope_cache(max_seq_len)

    def _build_rope_cache(self, max_seq_len: int):
        seq_idx = torch.arange(max_seq_len, dtype=self.theta.dtype, device=self.theta.device)
        idx_theta = torch.einsum("i,j->ij", seq_idx, self.theta).float()
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3)
        _batch, seq_len, num_heads, head_dim = x.shape
        rope_cache = self.cache[:seq_len].view(1, seq_len, num_heads, head_dim // 2, 2)
        x_shaped = x.float().reshape(*x.shape[:-1], head_dim // 2, 2)
        x_out = torch.stack(
            [
                x_shaped[..., 0] * rope_cache[..., 0] - x_shaped[..., 1] * rope_cache[..., 1],
                x_shaped[..., 1] * rope_cache[..., 0] + x_shaped[..., 0] * rope_cache[..., 1],
            ],
            dim=-1,
        )
        x_out = x_out.flatten(3).permute(0, 2, 1, 3)
        return x_out.type_as(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        max_seq_len: int = 64,
        qk_norm: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_drop = attn_drop
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = RotaryPosEmb(dim, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        qkv = self.qkv(x).reshape(
            batch_size, seq_len, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self.rope(self.q_norm(q))
        k = self.rope(self.k_norm(k))
        dropout_p = self.attn_drop if self.training else 0.0
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)
        x = x.transpose(1, 2).reshape(batch_size, seq_len, dim)
        return self.proj_drop(self.proj(x))


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.act(self.fc1(x)))
        return self.drop2(self.fc2(x))


class AdaLNBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=dropout, proj_drop=dropout)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), dropout=0.0)
        self.ada_ln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.dim = dim
        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.ada_ln[-1].weight, 0)
        nn.init.constant_(self.ada_ln[-1].bias, 0)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor):
        batch_size = x.shape[0]
        gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_ln(
            nn.SiLU()(t + c)
        ).view(batch_size, 6, 1, self.dim).unbind(1)

        x_norm1 = self.norm1(x).mul(scale1.add(1)).add_(shift1)
        x = x + self.attn(x_norm1).mul_(gamma1)

        x_norm2 = self.norm2(x).mul(scale2.add(1)).add_(shift2)
        x = x + self.mlp(x_norm2).mul_(gamma2)
        return x


class FlowTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        time_embed_dim: int = 256,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.Mish(),
            nn.Linear(time_embed_dim * 4, hidden_dim),
        )
        self.cond_embed = nn.Linear(condition_dim, hidden_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                AdaLNBlock(
                    dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)
        self._init_weights()

    def _init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        nn.init.normal_(self.time_embed[1].weight, std=0.02)
        nn.init.normal_(self.time_embed[3].weight, std=0.02)

    def forward(self, x: torch.Tensor, t: torch.Tensor, global_cond: torch.Tensor, local_cond=None):
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=x.device)
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        t = t.to(device=x.device, dtype=x.dtype)

        x = self.input_proj(x)
        t_embed = self.time_embed(t)
        c_embed = self.cond_embed(global_cond)
        for block in self.transformer_blocks:
            x = block(x, t_embed, c_embed)
        return self.out_proj(self.norm(x))
