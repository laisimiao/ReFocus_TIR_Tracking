import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from timm.models.layers import Mlp, DropPath, to_2tuple
from functools import reduce
from operator import mul


class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def forward(self, x):
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        return x


######################    Adapter    ########################
class AdapterBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, scale=0.5):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.attn_adapter = Adapter(dim, skip_connect=True)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp_adapter = Adapter(dim, skip_connect=False)
        self.scale = scale

    def forward(self, x, return_attention=False):
        if return_attention:
            feat, attn = self.attn(self.norm1(x), True)
            x = x + self.drop_path(self.attn_adapter(feat))
            xn = self.norm2(x)
            x = x + self.drop_path(self.mlp(xn)) + self.scale * self.mlp_adapter(xn)
            return x, attn
        else:
            x = x + self.drop_path(self.attn_adapter(self.attn(self.norm1(x))))
            xn = self.norm2(x)
            x = x + self.drop_path(self.mlp(xn)) + self.scale * self.mlp_adapter(xn)
            return x


######################    VPT    ########################
class VPTBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_tokens=5):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.vpt_prompt_embeddings = nn.Parameter(torch.zeros(1, num_tokens, dim))
        val = math.sqrt(6. / float(3 * reduce(mul, (16,16), 1) + dim))  # noqa 16->patch-size
        # xavier_uniform initialization
        nn.init.uniform_(self.vpt_prompt_embeddings.data, -val, val)


    def forward(self, x, return_attention=False):
        B, L = x.shape[0], x.shape[1]
        x = torch.cat([self.vpt_prompt_embeddings.expand(B, -1, -1), x], dim=1)
        if return_attention:
            feat, attn = self.attn(self.norm1(x), True)
            x = x + self.drop_path(feat)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x[:, -L:], attn
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x[:, -L:]


######################    LoRA    ########################
class LoRAAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 r=0, lora_alpha=1, lora_dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # LoRA codes
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        if r > 0:
            self.scaling = self.lora_alpha / self.r
            self.lora_A_q = nn.Parameter(self.qkv.weight.new_zeros((r, dim)))
            self.lora_B_q = nn.Parameter(self.qkv.weight.new_zeros((dim, r)))
            
            self.lora_A_v = nn.Parameter(self.qkv.weight.new_zeros((r, dim)))
            self.lora_B_v = nn.Parameter(self.qkv.weight.new_zeros((dim, r)))

            self.lora_A_o = nn.Parameter(self.proj.weight.new_zeros((r, dim)))
            self.lora_B_o = nn.Parameter(self.proj.weight.new_zeros((dim, r)))
    
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'lora_A_q'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A_q, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_A_v, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_A_o, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_q)
            nn.init.zeros_(self.lora_B_v)
            nn.init.zeros_(self.lora_B_o)


    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        qx = (self.lora_dropout(x) @ self.lora_A_q.transpose(0, 1) @ self.lora_B_q.transpose(0, 1)) * self.scaling
        vx = (self.lora_dropout(x) @ self.lora_A_v.transpose(0, 1) @ self.lora_B_v.transpose(0, 1)) * self.scaling
        qx = qx.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        vx = vx.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = q + qx
        v = v + vx

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        ox_prj = self.proj(x)
        ox_lora = (self.lora_dropout(x) @ self.lora_A_o.transpose(0, 1) @ self.lora_B_o.transpose(0, 1)) * self.scaling
        x = ox_prj + ox_lora
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        return x

class LoRAMlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.,
                 r=0, lora_alpha=1, lora_dropout=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])
        # LoRA codes
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        if r > 0:
            mlp_ratio = int(hidden_features // in_features)
            self.scaling = self.lora_alpha / self.r
            self.lora_A1 = nn.Parameter(self.fc1.weight.new_zeros((r, in_features)))
            self.lora_B1 = nn.Parameter(self.fc1.weight.new_zeros((hidden_features, r)))
            self.lora_A2 = nn.Parameter(self.fc2.weight.new_zeros((r*mlp_ratio, hidden_features)))
            self.lora_B2 = nn.Parameter(self.fc2.weight.new_zeros((out_features, r*mlp_ratio)))
        
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'lora_A1'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A1, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_A2, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B1)
            nn.init.zeros_(self.lora_B2)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = (self.lora_dropout(x) @ self.lora_A1.transpose(0, 1) @ self.lora_B1.transpose(0, 1)) * self.scaling
        x = x1 + x2
        x = self.act(x)
        x = self.drop1(x)
        x3 = self.fc2(x)
        x4 = (self.lora_dropout(x) @ self.lora_A2.transpose(0, 1) @ self.lora_B2.transpose(0, 1)) * self.scaling
        x = x3 + x4
        x = self.drop2(x)
        return x

class LoRABlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 r=0, lora_alpha=1, lora_dropout=0.):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = LoRAAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                                  r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = LoRAMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                           r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

    def forward(self, x, return_attention=False):
        if return_attention:
            feat, attn = self.attn(self.norm1(x), True)
            x = x + self.drop_path(feat)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, attn
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x


######################    ReFocus    ########################
class ReFocusAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, td=None, return_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        if td is not None:
            qkv_td = self.qkv(td).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            v = v + qkv_td[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        return x


class ReFocusBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = ReFocusAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, td=None, return_attention=False):
        if return_attention:
            feat, attn = self.attn(self.norm1(x), td=td, return_attention=True)
            x = x + self.drop_path(feat)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, attn
        else:
            x = x + self.drop_path(self.attn(self.norm1(x), td=td))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x