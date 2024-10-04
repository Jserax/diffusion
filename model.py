import math
from typing import Tuple

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from xformers.ops import memory_efficient_attention


class RelativePosEmb(nn.Module):
    def __init__(
        self,
        relative_bias_slope: float = 8.0,
        relative_bias_inter: int = 128,
        num_heads: int = 8,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.value = relative_bias_slope
        self.mlp = nn.Sequential(
            nn.Linear(2, relative_bias_inter, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(relative_bias_inter, num_heads, bias=False),
        )
        self.seq_len = 0
        self.rel_coords_table = None

    def forward(
        self,
        seq_len: int,
    ) -> torch.Tensor:
        if seq_len != self.seq_len:
            self.rel_coords_table = self._compute_rel_coords(seq_len)
            self.rel_coords_table = self.rel_coords_table.to(self.mlp[0].weight)
            self.seq_len = seq_len
        rel_bias = rearrange(self.mlp(self.rel_coords_table), "i j n -> 1 n i j")

        return rel_bias

    @torch.no_grad()
    def _compute_rel_coords(self, shape: Tuple[int, int]) -> torch.Tensor:
        h, w = shape
        coords_h = torch.linspace(-4.0 / 2, 4.0 / 2, h)
        coords_w = torch.linspace(-4.0 / 2, 4.0 / 2, w)

        rel_coords_h = (
            (coords_h[None, :] - coords_h[:, None])
            .repeat_interleave(w, dim=-1)
            .repeat_interleave(w, dim=-2)
        )
        rel_coords_w = (coords_w[None, :] - coords_w[:, None]).repeat((h, h))
        rel_coords = torch.stack((rel_coords_h, rel_coords_w))
        rel_coords_table = (
            torch.sign(rel_coords)
            * torch.log2(torch.abs(rel_coords) + 1.0)
            / math.log2(self.value + 1.0)
        )
        return rel_coords_table.permute(1, 2, 0)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        attn_dropout: float = 0.2,
        out_dropout: float = 0.2,
        attn_bias: bool = False,
        relative_bias_inter: int = 64,
        relative_bias_slope: float = 8.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=attn_bias)
        self.attn_dropout = attn_dropout
        self.out = nn.Linear(dim, dim, bias=attn_bias)
        self.out_dropout = nn.Dropout(out_dropout)
        # self.pos_emb = RelativePosEmb(
        #     relative_bias_slope=relative_bias_slope,
        #     relative_bias_inter=relative_bias_inter,
        #     num_heads=num_heads,
        # )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        B, L, C = x.size()
        H = W = int(L**0.5)
        q, k, v = rearrange(
            self.qkv(x), "b l (qkv h d) -> qkv b l h d", h=self.num_heads, qkv=3
        )
        # attn_bias = repeat(self.pos_emb((H, W)), "1 h i j -> k h i j", k=B).contiguous()

        x = memory_efficient_attention(
            q, k, v, p=self.attn_dropout if self.training else 0
        )
        # with torch.backends.cuda.sdp_kernel(
        #     enable_flash=False, enable_math=True, enable_mem_efficient=True
        # ):
        #     x = torch.nn.functional.scaled_dot_product_attention(
        #         q, k, v, attn_bias, dropout_p=self.attn_dropout if self.training else 0
        #     )
        x = rearrange(x, "b l h d -> b l (h d)")
        x = self.out_dropout(self.out(x))
        return x


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        cross_dim: int,
        num_heads: int = 8,
        attn_dropout: float = 0.2,
        out_dropout: float = 0.2,
        attn_bias: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q = nn.Linear(dim, dim, bias=attn_bias)
        self.kv = nn.Linear(cross_dim, 2 * dim, bias=attn_bias)
        self.attn_dropout = attn_dropout
        self.out = nn.Linear(dim, dim, bias=attn_bias)
        self.out_dropout = nn.Dropout(out_dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        q = rearrange(
            self.q(x),
            "b l (h d) -> b l h d",
            h=self.num_heads,
        )

        k, v = rearrange(
            self.kv(context), "b l (qkv h d) -> qkv b l h d", h=self.num_heads, qkv=2
        )
        x = memory_efficient_attention(
            q, k, v, p=self.attn_dropout if self.training else 0
        )
        # with torch.backends.cuda.sdp_kernel(
        #     enable_flash=False, enable_math=True, enable_mem_efficient=True
        # ):
        #     x = torch.nn.functional.scaled_dot_product_attention(
        #         q, k, v, dropout_p=self.attn_dropout if self.training else 0
        #     )
        x = rearrange(x, "b l h d -> b l (h d)")
        x = self.out_dropout(self.out(x))
        return x


class MLP(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        hid_dim: int = 512,
        bias: bool = False,
        dropout: float = 0.2,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.w1 = nn.Linear(dim, int(2 / 3 * hid_dim), bias=bias)
        self.w2 = nn.Linear(dim, int(2 / 3 * hid_dim), bias=bias)
        self.w3 = nn.Linear(int(2 / 3 * hid_dim), dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        swiglu = self.silu(self.w1(x)) * self.w2(x)
        swiglu = self.dropout(swiglu)
        return self.w3(swiglu)


class AttentionBlock(nn.Module):
    def __init__(
        self,
        heads: int = 8,
        emb_dim: int = 256,
        cross_dim: int = 256,
        mlp_hid_dim: int = 512,
        mlp_bias: bool = False,
        attn_bias: bool = False,
        attn_dropout: float = 0.0,
        attn_out_dropout: float = 0.0,
        attn_rel_bias_dim: int = 64,
        attn_rel_bias_slope: int = 8.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.norm_1 = nn.LayerNorm(emb_dim)
        self.attn = Attention(
            dim=emb_dim,
            num_heads=heads,
            attn_dropout=attn_dropout,
            out_dropout=attn_out_dropout,
            attn_bias=attn_bias,
            relative_bias_inter=attn_rel_bias_dim,
            relative_bias_slope=attn_rel_bias_slope,
        )

        self.norm_2 = nn.LayerNorm(emb_dim)
        self.cross_attn = CrossAttention(
            dim=emb_dim,
            cross_dim=cross_dim,
            num_heads=heads,
            attn_dropout=attn_dropout,
            out_dropout=attn_out_dropout,
            attn_bias=attn_bias,
        )

        self.norm_3 = nn.LayerNorm(emb_dim)
        self.mlp = MLP(emb_dim, mlp_hid_dim, mlp_bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        x = x + self.attn(self.norm_1(x))
        x = x + self.cross_attn(self.norm_2(x), context)
        x = x + self.mlp(self.norm_3(x))
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        return x


class Downsample(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.down = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, stride=2)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.down(x)


class Upsample(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.up = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.up(F.interpolate(x, scale_factor=2))


class ContextBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        context_dim: int,
        out_dim: int,
        groups: int = 32,
        bias: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.norm = nn.GroupNorm(groups, out_dim)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Sequential(
            nn.Linear(context_dim, context_dim, bias=bias),
            nn.LeakyReLU(0.2, True),
            nn.Linear(context_dim, 2 * out_dim, bias=bias),
        )
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        context = self.projection(context)
        context = self.pooling(context.permute(0, 2, 1)).unsqueeze(-1)
        scale, bias = context.chunk(2, dim=1)
        return scale * self.norm(self.conv(x)) + bias


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        time_dim: int,
        groups: int = 32,
        bias: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=bias)
        self.time_emb = nn.Linear(time_dim, 2 * out_dim)
        if in_dim == out_dim:
            self.skip_conv = nn.Identity()
        else:
            self.skip_conv = nn.Conv2d(
                in_dim,
                out_dim,
                kernel_size=1,
            )
        self.norm1 = nn.GroupNorm(groups, out_dim)
        self.norm2 = nn.GroupNorm(groups, out_dim)
        self.dropout = nn.Dropout2d(dropout)
        self.activation = nn.SiLU(inplace=False)

    def forward(
        self, x: torch.FloatTensor, time: torch.FloatTensor
    ) -> torch.FloatTensor:
        skip = self.skip_conv(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        time = self.activation(time)
        time_scale, time_bias = (
            self.time_emb(time).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        )
        x = time_scale * x + time_bias
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x + skip


class SwitchSequential(nn.Sequential):
    def forward(
        self, x: torch.FloatTensor, time: torch.FloatTensor, context: torch.FloatTensor
    ) -> torch.FloatTensor:
        for layer in self:
            if isinstance(layer, AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, ResidualBlock):
                x = layer(x, time)
            elif isinstance(layer, ContextBlock):
                x = layer(x, context)
            elif isinstance(layer, nn.Conv2d):
                x = layer(x)
        return x


class DiffusionModel(nn.Module):
    def __init__(
        self,
        start_dim: int = 64,
        num_layers: int = 3,
        num_steps: int = 1000,
        time_dim: int = 128,
        resblock_bias: bool = True,
        resblock_dropout: float = 0.2,
        context_dim: int = 312,
        context_block_bias: bool = True,
        attn_heads: int = 8,
        attn_hid_dim_mult: float = 2.0,
        attn_block_bias: bool = True,
        attn_dropout: float = 0.2,
        att_bias_dim: int = 64,
        attn_bias_slope: float = 8.0,
        groups: int = 32,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.time_emb = nn.Embedding(num_steps, time_dim)
        self.input = nn.Conv2d(3, start_dim, kernel_size=3, padding=1)
        self.downblocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()

        dim = start_dim
        for _ in range(num_layers):
            self.downblocks.append(
                SwitchSequential(
                    ContextBlock(
                        in_dim=dim,
                        context_dim=context_dim,
                        out_dim=dim,
                        groups=groups,
                        bias=context_block_bias,
                    ),
                    ResidualBlock(
                        in_dim=dim,
                        out_dim=dim,
                        time_dim=time_dim,
                        groups=groups,
                        bias=resblock_bias,
                        dropout=resblock_dropout,
                    ),
                    ResidualBlock(
                        in_dim=dim,
                        out_dim=dim,
                        time_dim=time_dim,
                        groups=groups,
                        bias=resblock_bias,
                        dropout=resblock_dropout,
                    ),
                    AttentionBlock(
                        heads=attn_heads,
                        emb_dim=dim,
                        cross_dim=context_dim,
                        mlp_hid_dim=int(attn_hid_dim_mult * dim),
                        mlp_bias=attn_block_bias,
                        attn_bias=attn_block_bias,
                        attn_dropout=attn_dropout,
                        attn_out_dropout=attn_dropout,
                        attn_rel_bias_dim=att_bias_dim,
                        attn_rel_bias_slope=attn_bias_slope,
                    ),
                )
            )
            self.downsample_blocks.append(
                Downsample(dim, 2 * dim),
            )
            dim *= 2
        self.mid = SwitchSequential(
            ContextBlock(
                in_dim=dim,
                context_dim=context_dim,
                out_dim=dim,
                groups=groups,
                bias=context_block_bias,
            ),
            ResidualBlock(
                in_dim=dim,
                out_dim=dim,
                time_dim=time_dim,
                groups=groups,
                bias=resblock_bias,
                dropout=resblock_dropout,
            ),
            ResidualBlock(
                in_dim=dim,
                out_dim=dim,
                time_dim=time_dim,
                groups=groups,
                bias=resblock_bias,
                dropout=resblock_dropout,
            ),
            AttentionBlock(
                heads=attn_heads,
                emb_dim=dim,
                cross_dim=context_dim,
                mlp_hid_dim=int(attn_hid_dim_mult * dim),
                mlp_bias=attn_block_bias,
                attn_bias=attn_block_bias,
                attn_dropout=attn_dropout,
                attn_out_dropout=attn_dropout,
                attn_rel_bias_dim=att_bias_dim,
                attn_rel_bias_slope=attn_bias_slope,
            ),
        )
        self.upblocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        for _ in range(num_layers):
            dim //= 2
            self.upsample_blocks.append(
                Upsample(2 * dim, dim),
            )
            self.upblocks.append(
                SwitchSequential(
                    ContextBlock(
                        in_dim=int(2 * dim),
                        context_dim=context_dim,
                        out_dim=dim,
                        groups=groups,
                        bias=context_block_bias,
                    ),
                    ResidualBlock(
                        in_dim=dim,
                        out_dim=dim,
                        time_dim=time_dim,
                        groups=groups,
                        bias=resblock_bias,
                        dropout=resblock_dropout,
                    ),
                    ResidualBlock(
                        in_dim=dim,
                        out_dim=dim,
                        time_dim=time_dim,
                        groups=groups,
                        bias=resblock_bias,
                        dropout=resblock_dropout,
                    ),
                    AttentionBlock(
                        heads=attn_heads,
                        emb_dim=dim,
                        cross_dim=context_dim,
                        mlp_hid_dim=int(attn_hid_dim_mult * dim),
                        mlp_bias=attn_block_bias,
                        attn_bias=attn_block_bias,
                        attn_dropout=attn_dropout,
                        attn_out_dropout=attn_dropout,
                        attn_rel_bias_dim=att_bias_dim,
                        attn_rel_bias_slope=attn_bias_slope,
                    ),
                ),
            )
        self.out = nn.Conv2d(dim, 3, kernel_size=3, padding=1)

    def params_count(self) -> int:
        return sum([p.numel() for p in self.parameters() if p.requires_grad])

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        time = self.time_emb(timestep)
        x = self.input(x)
        down_out = []
        for i, block in enumerate(self.downblocks):
            x = block(x, time, context)
            down_out.append(x)
            x = self.downsample_blocks[i](x)
        x = self.mid(x, time, context)
        for i, block in enumerate(self.upblocks):
            x = torch.cat((self.upsample_blocks[i](x), down_out.pop()), dim=1)
            x = block(x, time, context)
        return self.out(x)
