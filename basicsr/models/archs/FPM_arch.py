## Frequency-Guided State Space Model for Enhanced Remote Sensing Image Dehazing

import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange
import math
from typing import Optional, Callable
from einops import rearrange, repeat
from functools import partial


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):
        return self.cab(x)
class DynamicPosBias(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )

    def forward(self, biases):
        pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos

    def flops(self, N):
        flops = N * 2 * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.num_heads
        return flops


#########################################
class Attention(nn.Module):
    r""" Multi-head self attention module with dynamic position bias.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 position_bias=True):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.position_bias = position_bias
        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, H, W, mask=None):
        """
        Args:
            x: input features with shape of (num_groups*B, N, C)
            mask: (0/-inf) mask with shape of (num_groups, Gh*Gw, Gh*Gw) or None
            H: height of each group
            W: width of each group
        """
        group_size = (H, W)
        B_, N, C = x.shape
        assert H * W == N
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1).contiguous()  # (B_, self.num_heads, N, N), N = H*W

        if self.position_bias:
            # generate mother-set
            position_bias_h = torch.arange(1 - group_size[0], group_size[0], device=attn.device)
            position_bias_w = torch.arange(1 - group_size[1], group_size[1], device=attn.device)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))  # 2, 2Gh-1, 2W2-1
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()  # (2h-1)*(2w-1) 2

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(group_size[0], device=attn.device)
            coords_w = torch.arange(group_size[1], device=attn.device)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Gh, Gw
            coords_flatten = torch.flatten(coords, 1)  # 2, Gh*Gw
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Gh*Gw, Gh*Gw
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Gh*Gw, Gh*Gw, 2
            relative_coords[:, :, 0] += group_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += group_size[1] - 1
            relative_coords[:, :, 0] *= 2 * group_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Gh*Gw, Gh*Gw

            pos = self.pos(biases)  # 2Gh-1 * 2Gw-1, heads
            # select position bias
            relative_position_bias = pos[relative_position_index.view(-1)].view(
                group_size[0] * group_size[1], group_size[0] * group_size[1], -1)  # Gh*Gw,Gh*Gw,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Gh*Gw, Gh*Gw
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nP = mask.shape[0]
            attn = attn.view(B_ // nP, nP, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(
                0)  # (B, nP, nHead, N, N)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x




class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))


    def forward(self, input, x_size):
        # x [B,HW,C]
        B, C, H, W = input.shape
        input=input.permute(0,2,3,1)
        # input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        x = self.ln_1(input)
        x = input*self.skip_scale + self.drop_path(self.self_attention(x))
        x = x*self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x = x.permute(0, 3, 1, 2)
        # x = x.view(B, -1, C).contiguous()
        return x

##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x

########### window operation#############
def window_partition(x, win_size, dilation_rate=1):
    B, H, W, C = x.shape
    if dilation_rate !=1:
        x = x.permute(0,3,1,2) # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size,dilation=dilation_rate,padding=4*(dilation_rate-1),stride=win_size) # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0,2,1).contiguous().view(-1, C, win_size, win_size) # B' ,C ,Wh ,Ww
        windows = windows.permute(0,2,3,1).contiguous() # B' ,Wh ,Ww ,C
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C) # B' ,Wh ,Ww ,C
    return windows

def window_reverse(windows, win_size, H, W, dilation_rate=1):
    # B' ,Wh ,Ww ,C
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate !=1:
        x = windows.permute(0,5,3,4,1,2).contiguous() # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4*(dilation_rate-1),stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class lowFrequencyPromptFusion(nn.Module):
    def __init__(self, dim, dim_bak, num_heads,win_size=8, bias=False):
        super(lowFrequencyPromptFusion, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.ap_kv = nn.AdaptiveAvgPool2d(1)
        self.kv = nn.Conv2d(dim_bak, dim * 2, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d( dim, dim, kernel_size=1, bias=bias)

    def forward(self, feature, prompt_feature):
        b, c1,h,w = feature.shape
        _, c2,_,_ = prompt_feature.shape

        query = self.q(feature).reshape(b, h * w, self.num_heads, c1 // self.num_heads).permute(0, 2, 1, 3).contiguous()
        
        prompt_feature = self.ap_kv(prompt_feature)#.reshape(b, c2, -1).permute(0, 2, 1)
        key_value = self.kv(prompt_feature).reshape(b, 2*c1, -1).permute(0, 2, 1).contiguous().reshape(b, -1, 2, self.num_heads, c1 // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        key, value = key_value[0], key_value[1]

        attn = (query @ key.transpose(-2, -1).contiguous()) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ value)

        out = rearrange(out, 'b head (h w) c -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)

        return out

class LinearProjection(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., bias=True, isQuery = True):
        super().__init__()
        self.isQuery =isQuery
        inner_dim = dim_head *  heads
        self.heads = heads
        if self.isQuery:
            self.to_q = nn.Linear(dim, inner_dim, bias = bias)
        else:
            self.to_kv = nn.Linear(dim, 2*inner_dim, bias = bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B_,1,1)
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        if self.isQuery:
            q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4).contiguous()
            q = q[0]
            return q
        else:
            C = self.inner_dim 
            kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4).contiguous()
            k, v = kv[0], kv[1] 
            return k,v

class LinearProjection2(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., bias=True, isQuery = True):
        super().__init__()
        self.isQuery =isQuery
        inner_dim = dim_head *  heads
        self.heads = heads
        if self.isQuery:
            self.to_q = nn.Linear(dim, inner_dim, bias = bias)
        else:
            self.to_kv = nn.Linear(dim, inner_dim//2, bias = bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B_,1,1)
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        if self.isQuery:
            q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4).contiguous()
            q = q[0]
            return q
        else:
            C = self.inner_dim
            kv = self.to_kv(attn_kv).reshape(B_//4, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4).contiguous()
            k, v = kv[0], kv[1]
            return k,v

class LinearProjection3(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., bias=True, isQuery = True):
        super().__init__()
        self.isQuery =isQuery
        inner_dim = dim_head *  heads
        self.heads = heads
        if self.isQuery:
            self.to_q = nn.Linear(dim, inner_dim, bias = bias)
        else:
            self.to_kv = nn.Linear(dim, inner_dim//8, bias = bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        # print(x.shape)
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B_,1,1)
        else:
            attn_kv = x
        # print(attn_kv.shape)
        N_kv = attn_kv.size(1)
        if self.isQuery:
            q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4).contiguous()
            q = q[0]
            return q
        else:
            C = self.inner_dim
            kv = self.to_kv(attn_kv).reshape(B_//16, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4).contiguous()
            k, v = kv[0], kv[1]
            return k,v

class highFrequencyPromptFusion(nn.Module):
    def __init__(self, dim, dim_bak,win_size, num_heads, qkv_bias=True, qk_scale=None, bias=False):
        super(highFrequencyPromptFusion, self).__init__()
        self.num_heads = num_heads
        self.win_size = win_size  # Wh, Ww
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.to_q = LinearProjection(dim,num_heads,dim//num_heads,bias=qkv_bias,isQuery=True)
        self.to_q3 = LinearProjection3(dim, num_heads, dim // num_heads, bias=qkv_bias, isQuery=True)
        self.to_q2 = LinearProjection2(dim, num_heads, dim // num_heads, bias=qkv_bias, isQuery=True)

        self.to_kv = LinearProjection(dim_bak,num_heads,dim//num_heads,bias=qkv_bias,isQuery=False)
        self.to_kv3 = LinearProjection3(dim_bak, num_heads, dim // num_heads, bias=qkv_bias, isQuery=False)
        self.to_kv2 = LinearProjection2(dim_bak, num_heads, dim // num_heads, bias=qkv_bias, isQuery=False)

        self.kv_dwconv = nn.Conv2d(dim_bak , dim_bak, kernel_size=3, stride=1, padding=1, groups=dim_bak, bias=bias)
        
        self.softmax = nn.Softmax(dim=-1)

        self.project_out = nn.Linear(dim, dim)

    def forward(self, query_feature, key_value_feature):

        b,c,h,w = query_feature.shape
        _,c_2,h2,w2 = key_value_feature.shape
        # print(query_feature.shape)
        # print(key_value_feature.shape)
        key_value_feature = self.kv_dwconv(key_value_feature)
        
        # partition windows
        query_feature = rearrange(query_feature, ' b c1 h w -> b h w c1 ', h=h, w=w)
        query_feature_windows = window_partition(query_feature, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        query_feature_windows = query_feature_windows.view(-1, self.win_size * self.win_size, c)  # nW*B, win_size*win_size, C

        key_value_feature = rearrange(key_value_feature, ' b c2 h w -> b h w c2 ', h=h2, w=w2)
        key_value_feature_windows = window_partition(key_value_feature, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        key_value_feature_windows = key_value_feature_windows.view(-1, self.win_size * self.win_size, c_2)  # nW*B, win_size*win_size, C

        # print(query_feature_windows.shape)
        # print(key_value_feature_windows.shape)

        B_, N, C = query_feature_windows.shape
        query = self.to_q(query_feature_windows)
        query = query * self.scale
        # print(query.shape)

        # print(key.shape)
        attn = (query @ key.transpose(-2, -1).contiguous())
        attn = attn.softmax(dim=-1)

        out = (attn @ value).transpose(1, 2).contiguous().reshape(B_, N, C)

        out = self.project_out(out)

        # merge windows
        attn_windows = out.view(-1, self.win_size, self.win_size, C)
        attn_windows = window_reverse(attn_windows, self.win_size, h, w)  # B H' W' C
        return rearrange(attn_windows, 'b h w c -> b c h w', h=h, w=w)

##########################################################################
## channel dynamic filters
class dynamic_filter_channel(nn.Module):
    def __init__(self, inchannels, kernel_size=3, stride=1, group=8):
        super(dynamic_filter_channel, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group

        self.conv = nn.Conv2d(inchannels, group*kernel_size**2, kernel_size=1, stride=1, bias=False)
        self.conv_gate = nn.Conv2d(group*kernel_size**2, group*kernel_size**2, kernel_size=1, stride=1, bias=False)
        self.act_gate  = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(group*kernel_size**2)
        self.act = nn.Softmax(dim=-2)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

        self.pad = nn.ReflectionPad2d(kernel_size//2)

        self.ap_1 = nn.AdaptiveAvgPool2d((1, 1))
        #self.ap_2 = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x):
        identity_input = x 
        low_filter1 = self.ap_1(x)
        #low_filter2 = self.ap_2(x)
        low_filter = self.conv(low_filter1)
        low_filter = low_filter * self.act_gate(self.conv_gate(low_filter))
        low_filter = self.bn(low_filter)     

        n, c, h, w = x.shape  
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape(n, self.group, c//self.group, self.kernel_size**2, h*w)

        n,c1,p,q = low_filter.shape
        low_filter = low_filter.reshape(n, c1//self.kernel_size**2, self.kernel_size**2, p*q).unsqueeze(2)
       
        low_filter = self.act(low_filter)
        # print('low_filter size',low_filter.shape)
        # print('low_filter n,c1,p,q',n,c1,p,q)
    
        low_part = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)

        out_high = identity_input - low_part
        return low_part, out_high


class frequenctSpecificPromptGenetator(nn.Module):
    def __init__(self, dim=3,h=128,w=65, flag_highF=True):
        super().__init__()
        self.flag_highF = flag_highF
        k_size = 3
        if flag_highF:
            w = (w - 1) * 2
            self.w = w
            self.h = h
            self.weight = nn.Parameter(torch.randn(1,dim, h, w, dtype=torch.float32) * 0.02)
            self.body = nn.Sequential(nn.Conv2d(dim, dim, (1,k_size), padding=(0, k_size//2), groups=dim),
                                      nn.Conv2d(dim, dim, (k_size,1), padding=(k_size//2, 0), groups=dim),
                                      nn.GELU())
        else:
            self.complex_weight = nn.Parameter(torch.randn(1,dim, h, w, 2, dtype=torch.float32) * 0.02)
            self.body = nn.Sequential(nn.Conv2d(2*dim,2*dim,kernel_size=1,stride=1),
                                    nn.GELU(),
                                    )
            

    def forward(self, ffm, H, W):
        if self.flag_highF:
            ffm = F.interpolate(ffm, size=(H, W), mode='bilinear')
            y_att = self.body(ffm)

            y_f = y_att * ffm
            y = y_f * self.weight

        else:
            ffm = F.interpolate(ffm, size=(H, W), mode='bicubic')
            y = torch.fft.rfft2(ffm.to(torch.float32).cuda())
            y_imag = y.imag
            y_real = y.real
            y_f = torch.cat([y_real, y_imag], dim=1)
            weight = torch.complex(self.complex_weight[..., 0],self.complex_weight[..., 1])
            y_att = self.body(y_f)
            y_f = y_f * y_att
            y_real, y_imag = torch.chunk(y_f, 2, dim=1)
            y = torch.complex(y_real, y_imag)
            y = y * weight
            y = torch.fft.irfft2(y, s=(H, W))
        
        return y
    
##########################################################################
## PromptModule
class PromptModule(nn.Module):
    def __init__(self, basic_dim=32, dim=32, input_resolution=128):
        super().__init__()
        h = input_resolution
        w = input_resolution//2 +1
        self.simple_Fusion = nn.Conv2d(2*dim,dim,kernel_size=1,stride=1)

        self.FSPG_high = frequenctSpecificPromptGenetator(basic_dim,h,w, flag_highF=True)
        self.FSPG_low = frequenctSpecificPromptGenetator(basic_dim,h,w, flag_highF=False)


        self.modulator_hi = highFrequencyPromptFusion(dim, basic_dim, win_size=8, num_heads=2, bias=False)
        self.modulator_lo = lowFrequencyPromptFusion(dim, basic_dim, win_size=8, num_heads=2, bias=False)
    def forward(self, low_part, out_high , x):
        b,c,h,w = x.shape
        b,c_h,h_h,w_h=out_high.shape
        b,c_l,h_l,w_l=low_part.shape

        y_h = self.FSPG_high(out_high, h_h, w_h)
        y_l = self.FSPG_low(low_part, h_l, w_l)

        y_h = self.modulator_hi(x,y_h)
        y_l = self.modulator_lo(x,y_l)

        x = self.simple_Fusion(torch.cat([y_h,y_l], dim=1)) 

        return x

## PromptModule
class splitFrequencyModule(nn.Module):
    def __init__(self, basic_dim=32, dim=32, input_resolution=128):
        super().__init__()

        self.dyna_channel = dynamic_filter_channel(inchannels=basic_dim)
    def forward(self, F_low ):
        _,c_basic,h_ori, w_ori = F_low.shape

        low_part, out_high = self.dyna_channel(F_low)

        return low_part, out_high


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
##---------- FPM -----------------------
class FPM(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8],
        drop_path_rate=0.,
        mlp_ratio=2.,
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        # ffn_expansion_factor = 2.66,
        bias = False,
        # LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(FPM, self).__init__()
        self.mlp_ratio = mlp_ratio
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        base_d_state = 4
        self.encoder_level1 = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=base_d_state* 2 ** 2,
            )
            for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 1),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 2),
            )
            for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 2),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 2),
            )
            for i in range(num_blocks[2])])

        self.splitFre =splitFrequencyModule(basic_dim= dim,dim=int(dim*2**2),input_resolution=32)
        self.decoder_level3 = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 2),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 2),
            )
            for i in range(num_blocks[2])])
        self.prompt_d3 = PromptModule(basic_dim= dim,dim=int(dim*2**2),input_resolution=256)

        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 1),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 2),
            )
            for i in range(num_blocks[1])])

        self.prompt_d2 = PromptModule(basic_dim= dim,dim=int(dim*2**1),input_resolution=256)

        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 1),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 2),
            )
            for i in range(num_blocks[0])])

        self.prompt_d1 = PromptModule(basic_dim= dim,dim=int(dim*2**1),input_resolution=256)

        self.refinement =nn.ModuleList([
            VSSBlock(
                hidden_dim=int(dim * 2 ** 1),
                drop_path=drop_path_rate,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                expand=self.mlp_ratio,
                d_state=int(base_d_state * 2 ** 2),
            )
            for i in range(num_refinement_blocks)])

        self.prompt_r = PromptModule(basic_dim= dim,dim=int(dim*2**1),input_resolution=256)
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        _, _, H, W = inp_img.shape
        inp_enc_level1 = self.patch_embed(inp_img)
        for layer in self.encoder_level1:
            out_enc_level1 = layer(inp_enc_level1, [H, W])
        inp_enc_level2 = self.down1_2(out_enc_level1)
        for layer in self.encoder_level2:
            out_enc_level2 = layer(inp_enc_level2, [H // 2, W // 2])

        inp_enc_level3 = self.down2_3(out_enc_level2)
        for layer in self.encoder_level3:
            out_enc_level3 = layer(inp_enc_level3, [H // 4, W // 4])

        for layer in self.decoder_level3:
            out_dec_level3 = layer(out_enc_level3, [H // 4, W // 4])

        # Frequency - Decoupled Prompting Block
        low_part, out_high = self.splitFre(inp_enc_level1)
        out_dec_level3 = self.prompt_d3(low_part, out_high, out_dec_level3) + out_dec_level3
        # print(out_dec_level3.shape)
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        for layer in self.decoder_level2:
            out_dec_level2 = layer(inp_dec_level2, [H // 2, W // 2])
        out_dec_level2 = self.prompt_d2(low_part, out_high, out_dec_level2) + out_dec_level2
        # print(out_dec_level2.shape)
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        for layer in self.decoder_level1:
            out_dec_level1 = layer(inp_dec_level1, [H, W])
        # print(out_dec_level1.shape)
        out_dec_level1 = self.prompt_d1(low_part, out_high, out_dec_level1) + out_dec_level1

        for layer in self.refinement:
            out_dec_level1 = layer(out_dec_level1, [H, W])
        out_dec_level1 = self.prompt_r(low_part, out_high,out_dec_level1) + out_dec_level1

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img


        return out_dec_level1

