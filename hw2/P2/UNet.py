import math
from typing import List

import torch
from torch import nn
from torch.nn import functional as F


swish = F.silu


@torch.no_grad()
def variance_scaling_init_(tensor, scale=1, mode="fan_avg", distribution="uniform"):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)

    if mode == "fan_in":
        scale /= fan_in

    elif mode == "fan_out":
        scale /= fan_out

    else:
        scale /= (fan_in + fan_out) / 2

    if distribution == "normal":
        std = math.sqrt(scale)

        return tensor.normal_(0, std)

    else:
        bound = math.sqrt(3 * scale)

        return tensor.uniform_(-bound, bound)


def conv2d(
    in_channel,
    out_channel,
    kernel_size,
    stride=1,
    padding=0,
    bias=True,
    scale=1,
    mode="fan_avg",
):
    conv = nn.Conv2d(
        in_channel, out_channel, kernel_size, stride=stride, padding=padding, bias=bias
    )

    variance_scaling_init_(conv.weight, scale, mode=mode)

    if bias:
        nn.init.zeros_(conv.bias)

    return conv


def linear(in_channel, out_channel, scale=1, mode="fan_avg"):
    lin = nn.Linear(in_channel, out_channel)

    variance_scaling_init_(lin.weight, scale, mode=mode)
    nn.init.zeros_(lin.bias)

    return lin


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return swish(input)


class Upsample(nn.Sequential):
    def __init__(self, channel):
        layers = [
            nn.Upsample(scale_factor=2, mode="nearest"),
            conv2d(channel, channel, 3, padding=1),
        ]

        super().__init__(*layers)


class Downsample(nn.Sequential):
    def __init__(self, channel):
        layers = [conv2d(channel, channel, 3, stride=2, padding=1)]

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(
        self, in_channel, out_channel, time_dim, use_affine_time=False, dropout=0
    ):
        super().__init__()

        self.use_affine_time = use_affine_time
        time_out_dim = out_channel
        time_scale = 1
        norm_affine = True

        if self.use_affine_time:
            time_out_dim *= 2
            time_scale = 1e-10
            norm_affine = False

        self.norm1 = nn.GroupNorm(32, in_channel)
        self.activation1 = Swish()
        self.conv1 = conv2d(in_channel, out_channel, 3, padding=1)

        self.time = nn.Sequential(
            Swish(), linear(time_dim, time_out_dim, scale=time_scale)
        )

        self.norm2 = nn.GroupNorm(32, out_channel, affine=norm_affine)
        self.activation2 = Swish()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = conv2d(out_channel, out_channel, 3, padding=1, scale=1e-10)

        if in_channel != out_channel:
            self.skip = conv2d(in_channel, out_channel, 1)

        else:
            self.skip = None

    def forward(self, input, time):
        batch = input.shape[0]

        out = self.conv1(self.activation1(self.norm1(input)))

        if self.use_affine_time:
            gamma, beta = self.time(time).view(batch, -1, 1, 1).chunk(2, dim=1)
            out = (1 + gamma) * self.norm2(out) + beta

        else:
            out = out + self.time(time).view(batch, -1, 1, 1)
            out = self.norm2(out)

        out = self.conv2(self.dropout(self.activation2(out)))

        if self.skip is not None:
            input = self.skip(input)

        return out + input


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(32, in_channel)
        self.qkv = conv2d(in_channel, in_channel * 3, 1)
        self.out = conv2d(in_channel, in_channel, 1, scale=1e-10)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim

        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000) / dim)
        )

        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        shape = input.shape
        sinusoid_in = torch.ger(input.view(-1).float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)

        return pos_emb


class ResBlockWithAttention(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        time_dim,
        dropout,
        use_attention=False,
        attention_head=1,
        use_affine_time=False,
    ):
        super().__init__()

        self.resblocks = ResBlock(
            in_channel, out_channel, time_dim, use_affine_time, dropout
        )

        if use_attention:
            self.attention = SelfAttention(out_channel, n_head=attention_head)

        else:
            self.attention = None

    def forward(self, input, time):
        out = self.resblocks(input, time)

        if self.attention is not None:
            out = self.attention(out)

        return out


def spatial_unfold(input, unfold):
    if unfold == 1:
        return input

    batch, channel, height, width = input.shape
    h_unfold = height * unfold
    w_unfold = width * unfold

    return (
        input.view(batch, -1, unfold, unfold, height, width)
        .permute(0, 1, 4, 2, 5, 3)
        .reshape(batch, -1, h_unfold, w_unfold)
    )


class UNet(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128,
        attn_heads=1,
        use_affine_time=False,
        dropout=0,
    ):
        super().__init__()

        time_dim = channel * 4

        self.time = nn.Sequential(
            TimeEmbedding(channel),
            linear(channel, time_dim),
            Swish(),
            linear(time_dim, time_dim),
        )

        self.down1 = conv2d(in_channel, channel, 3, padding=1)
        self.down2 = ResBlockWithAttention(
            128,
            128,
            time_dim,
            dropout,
            use_attention=False,
            attention_head=attn_heads,
            use_affine_time=use_affine_time,
        )
        self.down3 = ResBlockWithAttention(
            128,
            128,
            time_dim,
            dropout,
            use_attention=False,
            attention_head=attn_heads,
            use_affine_time=use_affine_time,
        )
        self.down4 = Downsample(128)
        self.down5 = ResBlockWithAttention(
            128,
            128,
            time_dim,
            dropout,
            use_attention=False,
            attention_head=attn_heads,
            use_affine_time=use_affine_time,
        )
        self.down6 = ResBlockWithAttention(
            128,
            128,
            time_dim,
            dropout,
            use_attention=False,
            attention_head=attn_heads,
            use_affine_time=use_affine_time,
        )
        self.down7 = Downsample(128)
        self.down8 = ResBlockWithAttention(
            128,
            256,
            time_dim,
            dropout,
            use_attention=False,
            attention_head=attn_heads,
            use_affine_time=use_affine_time,
        )
        self.down9 = ResBlockWithAttention(
            256,
            256,
            time_dim,
            dropout,
            use_attention=False,
            attention_head=attn_heads,
            use_affine_time=use_affine_time,
        )
        self.down10 = Downsample(256)
        self.down11 = ResBlockWithAttention(
            256,
            256,
            time_dim,
            dropout,
            use_attention=False,
            attention_head=attn_heads,
            use_affine_time=use_affine_time,
        )
        self.down12 = ResBlockWithAttention(
            256,
            256,
            time_dim,
            dropout,
            use_attention=False,
            attention_head=attn_heads,
            use_affine_time=use_affine_time,
        )
        self.down13 = Downsample(256)
        self.down14 = ResBlockWithAttention(
            256,
            512,
            time_dim,
            dropout,
            use_attention=True,
            attention_head=attn_heads,
            use_affine_time=use_affine_time,
        )
        self.down15 = ResBlockWithAttention(
            512,
            512,
            time_dim,
            dropout,
            use_attention=True,
            attention_head=attn_heads,
            use_affine_time=use_affine_time,
        )
        self.down16 = Downsample(512)
        self.down17 = ResBlockWithAttention(
            512,
            512,
            time_dim,
            dropout,
            use_attention=False,
            attention_head=attn_heads,
            use_affine_time=use_affine_time,
        )
        self.down18 = ResBlockWithAttention(
            512,
            512,
            time_dim,
            dropout,
            use_attention=False,
            attention_head=attn_heads,
            use_affine_time=use_affine_time,
        )

        self.mid1 = ResBlockWithAttention(
            512,
            512,
            time_dim,
            dropout=dropout,
            use_attention=True,
            attention_head=attn_heads,
            use_affine_time=use_affine_time,
        )
        self.mid2 = ResBlockWithAttention(
            512,
            512,
            time_dim,
            dropout=dropout,
            use_affine_time=use_affine_time,
        )

        self.up1 = ResBlockWithAttention(
            1024,
            512,
            time_dim,
            dropout,
            use_attention=False,
            attention_head=attn_heads,
            use_affine_time=use_affine_time,
        )
        self.up2 = ResBlockWithAttention(
            1024,
            512,
            time_dim,
            dropout,
            use_attention=False,
            attention_head=attn_heads,
            use_affine_time=use_affine_time,
        )
        self.up3 = ResBlockWithAttention(
            1024,
            512,
            time_dim,
            dropout,
            use_attention=False,
            attention_head=attn_heads,
            use_affine_time=use_affine_time,
        )
        self.up4 = Upsample(512)
        self.up5 = ResBlockWithAttention(
            1024,
            512,
            time_dim,
            dropout,
            use_attention=True,
            attention_head=attn_heads,
            use_affine_time=use_affine_time,
        )
        self.up6 = ResBlockWithAttention(
            1024,
            512,
            time_dim,
            dropout,
            use_attention=True,
            attention_head=attn_heads,
            use_affine_time=use_affine_time,
        )
        self.up7 = ResBlockWithAttention(
            768,
            512,
            time_dim,
            dropout,
            use_attention=True,
            attention_head=attn_heads,
            use_affine_time=use_affine_time,
        )
        self.up8 = Upsample(512)
        self.up9 = ResBlockWithAttention(
            768,
            256,
            time_dim,
            dropout,
            use_attention=False,
            attention_head=attn_heads,
            use_affine_time=use_affine_time,
        )
        self.up10 = ResBlockWithAttention(
            512,
            256,
            time_dim,
            dropout,
            use_attention=False,
            attention_head=attn_heads,
            use_affine_time=use_affine_time,
        )
        self.up11 = ResBlockWithAttention(
            512,
            256,
            time_dim,
            dropout,
            use_attention=False,
            attention_head=attn_heads,
            use_affine_time=use_affine_time,
        )
        self.up12 = Upsample(256)
        self.up13 = ResBlockWithAttention(
            512,
            256,
            time_dim,
            dropout,
            use_attention=False,
            attention_head=attn_heads,
            use_affine_time=use_affine_time,
        )
        self.up14 = ResBlockWithAttention(
            512,
            256,
            time_dim,
            dropout,
            use_attention=False,
            attention_head=attn_heads,
            use_affine_time=use_affine_time,
        )
        self.up15 = ResBlockWithAttention(
            384,
            256,
            time_dim,
            dropout,
            use_attention=False,
            attention_head=attn_heads,
            use_affine_time=use_affine_time,
        )
        self.up16 = Upsample(256)
        self.up17 = ResBlockWithAttention(
            384,
            128,
            time_dim,
            dropout,
            use_attention=False,
            attention_head=attn_heads,
            use_affine_time=use_affine_time,
        )
        self.up18 = ResBlockWithAttention(
            256,
            128,
            time_dim,
            dropout,
            use_attention=False,
            attention_head=attn_heads,
            use_affine_time=use_affine_time,
        )
        self.up19 = ResBlockWithAttention(
            256,
            128,
            time_dim,
            dropout,
            use_attention=False,
            attention_head=attn_heads,
            use_affine_time=use_affine_time,
        )
        self.up20 = Upsample(128)
        self.up21 = ResBlockWithAttention(
            256,
            128,
            time_dim,
            dropout,
            use_attention=False,
            attention_head=attn_heads,
            use_affine_time=use_affine_time,
        )
        self.up22 = ResBlockWithAttention(
            256,
            128,
            time_dim,
            dropout,
            use_attention=False,
            attention_head=attn_heads,
            use_affine_time=use_affine_time,
        )
        self.up23 = ResBlockWithAttention(
            256,
            128,
            time_dim,
            dropout,
            use_attention=False,
            attention_head=attn_heads,
            use_affine_time=use_affine_time,
        )

        self.out = nn.Sequential(
            nn.GroupNorm(32, 128),
            Swish(),
            conv2d(128, 3, 3, padding=1, scale=1e-10),
        )

    def forward(self, x, time):
        time_embed = self.time(time)

        feats = []

        x = self.down1(x)
        feats.append(x)
        x = self.down2(x, time_embed)
        feats.append(x)
        x = self.down3(x, time_embed)
        feats.append(x)
        x = self.down4(x)
        feats.append(x)
        x = self.down5(x, time_embed)
        feats.append(x)
        x = self.down6(x, time_embed)
        feats.append(x)
        x = self.down7(x)
        feats.append(x)
        x = self.down8(x, time_embed)
        feats.append(x)
        x = self.down9(x, time_embed)
        feats.append(x)
        x = self.down10(x)
        feats.append(x)
        x = self.down11(x, time_embed)
        feats.append(x)
        x = self.down12(x, time_embed)
        feats.append(x)
        x = self.down13(x)
        feats.append(x)
        x = self.down14(x, time_embed)
        feats.append(x)
        x = self.down15(x, time_embed)
        feats.append(x)
        x = self.down16(x)
        feats.append(x)
        x = self.down17(x, time_embed)
        feats.append(x)
        x = self.down18(x, time_embed)
        feats.append(x)

        x = self.mid1(x, time_embed)
        x = self.mid2(x, time_embed)

        x = self.up1(torch.cat((x, feats.pop()), 1), time_embed)
        x = self.up2(torch.cat((x, feats.pop()), 1), time_embed)
        x = self.up3(torch.cat((x, feats.pop()), 1), time_embed)
        x = self.up4(x)
        x = self.up5(torch.cat((x, feats.pop()), 1), time_embed)
        x = self.up6(torch.cat((x, feats.pop()), 1), time_embed)
        x = self.up7(torch.cat((x, feats.pop()), 1), time_embed)
        x = self.up8(x)
        x = self.up9(torch.cat((x, feats.pop()), 1), time_embed)
        x = self.up10(torch.cat((x, feats.pop()), 1), time_embed)
        x = self.up11(torch.cat((x, feats.pop()), 1), time_embed)
        x = self.up12(x)
        x = self.up13(torch.cat((x, feats.pop()), 1), time_embed)
        x = self.up14(torch.cat((x, feats.pop()), 1), time_embed)
        x = self.up15(torch.cat((x, feats.pop()), 1), time_embed)
        x = self.up16(x)
        x = self.up17(torch.cat((x, feats.pop()), 1), time_embed)
        x = self.up18(torch.cat((x, feats.pop()), 1), time_embed)
        x = self.up19(torch.cat((x, feats.pop()), 1), time_embed)
        x = self.up20(x)
        x = self.up21(torch.cat((x, feats.pop()), 1), time_embed)
        x = self.up22(torch.cat((x, feats.pop()), 1), time_embed)
        x = self.up23(torch.cat((x, feats.pop()), 1), time_embed)

        out = self.out(x)
        out = spatial_unfold(out, 1)

        return out
