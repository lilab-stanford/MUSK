import torch
import torch.nn as nn
from timm.models.layers import Mlp, DropPath
import torch.nn.functional as F
from typing import Optional

try:
    from torch.nn.functional import scaled_dot_product_attention

    fused_attn = True
except:
    fused_attn = False


class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            tau: float = 1,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.fused_attn = fused_attn  # for torch2.0
        self.tau = tau

        self.to_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        B_c, N_c, C_c = context.shape

        assert B_c == B and C == C_c, "query and context must be the same batch size and dimension"

        # process k, v
        kv = self.to_kv(context).reshape(B_c, N_c, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        # process q
        q = self.to_q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            # Apply temperature scaling to the query and key dot products
            q = q / self.tau
            k = k / self.tau

            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            # attn = q @ k.transpose(-2, -1)
            attn = (q @ k.transpose(-2, -1)) / self.tau  # Apply temperature scaling here
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class AttnBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm_context = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, context: torch.Tensor, skip_connect=True) -> torch.Tensor:
        # attention
        if skip_connect:
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), self.norm_context(context))))
        else:
            x = self.drop_path1(self.ls1(self.attn(self.norm1(x), self.norm_context(context))))
            
        # mlp
        if skip_connect:
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        else:
            x = self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        return x


if __name__ == "__main__":
    embed_dim = 1024
    num_blocks = 4
    vl_decoder = torch.nn.ModuleList([AttnBlock(embed_dim, 16) for _ in range(num_blocks)])
    total_params = sum([p.numel() for p in vl_decoder.parameters()])
    print(total_params/1e6)


