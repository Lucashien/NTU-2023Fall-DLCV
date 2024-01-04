import math
import collections
import torch
from torch import nn, Tensor
import torch.nn.functional as F

PAD_TOKEN = 50256
BOS_TOKEN = 50256
EOS_TOKEN = 50256
device = "cuda" if torch.cuda.is_available() else "cpu"


class Config:
    def __init__(self, checkpoint=None):
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        self.vocab_size = 50257
        self.block_size = 1024
        self.checkpoint = checkpoint


# self attention layer
class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        size = cfg.block_size
        self.register_buffer(
            "bias", torch.tril(torch.ones(size, size)).view(1, 1, size, size)
        )

    def forward(self, x):
        B, T, C = x.size()  # batch, context, embedding
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.crossattn = CrossAttention(cfg)  # Cross Attention
        # multi-layer perceptron
        self.mlp = nn.Sequential(
            collections.OrderedDict(
                [
                    ("c_fc", nn.Linear(cfg.n_embd, 4 * cfg.n_embd)),
                    ("act", nn.GELU(approximate="tanh")),
                    ("c_proj", nn.Linear(4 * cfg.n_embd, cfg.n_embd)),
                ]
            )
        )

    def forward(self, x, encoder_output):
        x = x + self.attn(self.ln_1(x))
        x = x + self.crossattn(x, encoder_output)
        x = x + self.mlp(self.ln_2(x))
        return x


class CrossAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.query = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.key = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.value = nn.Linear(cfg.n_embd, cfg.n_embd)

        # 這是Multi-head Attention的頭數，即將注意力機制分割為多個小部分的數量
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd  # embedding的維度
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)  # 線性變換，用於在計算完注意力後對結果進行變換

    # input: (id that transform by captions, encoder feature)
    def forward(self, x, encoder_output):
        # Batch, Sequence lengh, Feature dimension
        # x.shape = [B,T,C]
        B, T, C = x.size()

        _, S, _ = encoder_output.size()  # (64,197,768)

        # q為decoder的線性變換(Linear)結果 -> view 把 C維根據n_head切開 # transopse交換[1][2]的tensor
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = (
            self.key(encoder_output)
            .view(B, S, self.n_head, C // self.n_head)  # (1, 197, 12, 64)
            .transpose(1, 2)
        )
        v = (
            self.value(encoder_output)
            .view(B, S, self.n_head, C // self.n_head)
            .transpose(1, 2)
        )

        # core calc
        # (q dot k)/k_dim_sqrt
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)

        return self.c_proj(y)


class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),  # (50257,768)
                wpe=nn.Embedding(cfg.block_size, cfg.n_embd),  # (1024,768)
                h=nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)]),
                ln_f=nn.LayerNorm(cfg.n_embd),
            )
        )
        # Language Model head -> 預測單字的機率分佈
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # load checkpoint
        if self.cfg.checkpoint is not None:
            state_dict = torch.load(self.cfg.checkpoint)
            transposed = [".c_attn.weight", ".c_fc.weight", ".c_proj.weight"]
            for key, value in state_dict.items():
                if any(key.endswith(w) for w in transposed):
                    state_dict[key] = value.t()
            self.transformer.load_state_dict(state_dict, strict=False)

    # x is id that transform by captions
    def forward(self, x: Tensor, encoder_feature: Tensor):
        # narrow (dim,start,len)
        x = torch.narrow(x, 1, 0, min(x.size(1), self.block_size))
        pos = torch.arange(x.size()[1], dtype=torch.long, device=x.device).unsqueeze(0)

        # (text) word token embedding + word position embedding
        x = self.transformer.wte(x) + self.transformer.wpe(pos)
        # Encoder output to across attention layer in block
        for block in self.transformer.h:
            x = block(x, encoder_feature)

        # Generator
        x = self.lm_head(self.transformer.ln_f(x))

        return x
