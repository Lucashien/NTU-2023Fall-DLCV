import json, os
import torch
from torch import Tensor, nn
from PIL import Image
from tqdm.auto import tqdm
import torch.nn.functional as F
from tokenizer import BPETokenizer
import timm
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import math
import collections
from P2_adapter import Attention, Config
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {device}")
PAD_TOKEN = 50256
UNK_TOKEN = 1
BOS_TOKEN = 50256
EOS_TOKEN = 50256


class getDataset(Dataset):
    def __init__(self, img_dir, transform):
        super().__init__()
        self.img_dir = Path(img_dir)
        self.Transform = transform
        self.Image_names = [p for p in self.img_dir.glob("*")]

    def __getitem__(self, idx):
        img = Image.open(self.Image_names[idx]).convert("RGB")
        ori_trans = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        ori_img = ori_trans(img)
        img = self.Transform(img)
        return (
            ori_img,
            img,
            os.path.splitext(os.path.basename(self.Image_names[idx]))[0],
        )

    def __len__(self):
        return len(self.Image_names)


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

        self.adapter_layer1 = nn.Sequential(
            nn.Linear(cfg.n_embd, 128),
            nn.GELU(approximate="tanh"),
            nn.Linear(128, cfg.n_embd),
        )
        self.adapter_layer2 = nn.Sequential(
            nn.Linear(cfg.n_embd, 128),
            nn.GELU(approximate="tanh"),
            nn.Linear(128, cfg.n_embd),
        )

    def forward(self, x, encoder_output):
        x = x + self.attn(self.ln_1(x))
        x = x + self.adapter_layer1(x)
        cross_att_output, _, _ = self.crossattn(x, encoder_output)
        x = x + cross_att_output
        x = x + self.mlp(self.ln_2(x))
        x = x + self.adapter_layer2(x)
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
        query = self.query(x)  # [B, tgt Sequence lengh, Feature dimension]
        key = self.key(encoder_output)  # [B, source sequence length, Feature dimension]
        print((query @ key.transpose(-2,-1)).shape)
        q = query.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = key.view(B, S, self.n_head, C // self.n_head).transpose(  # (1, 257, 12, 64)
            1, 2
        )
        v = (
            self.value(encoder_output)
            .view(B, S, self.n_head, C // self.n_head)
            .transpose(1, 2)
        )

        # core calc
        # (q dot k)/
        # print("in ",q.shape,k.transpose(-2, -1).shape)
        # print("in ",(q @ k.transpose(-2, -1)).shape)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)

        return self.c_proj(y), query, key


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


class ImgCaptionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Tokenizer
        self.tokenizer = BPETokenizer()

        # Encoder (pretrained ViT model)
        self.encoder = timm.create_model(
            "vit_large_patch14_clip_224.openai_ft_in12k_in1k",
            pretrained=True,
            num_classes=0,
        ).to(device)
        self.feature_resize = nn.Linear(1024, 768)

        # Decoder
        self.cfg = Config("../hw3_data/p2_data/decoder_model.bin")
        self.decoder = Decoder(self.cfg).to(device)

        self.test = ""
        # Loss
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.loss = 0

    def forward(self, batch_imgs, batch_captions, batch_attmask):
        ground_truth = torch.concat(
            (batch_captions[:, 1:], batch_captions[:, :1]), dim=1
        )
        batch_attmask = torch.concat(
            (
                batch_attmask[:, 1:],
                torch.zeros(
                    (batch_attmask.shape[0], 1),
                    dtype=batch_attmask.dtype,
                    device=batch_attmask.device,
                ),
            ),
            dim=1,
        )
        feature = self.encoder.forward_features(batch_imgs)  # feature [64, 197, 768]
        feature = self.feature_resize(feature)
        decoder_output = self.decoder(batch_captions, feature)

        # setting ground truth
        for i, attmask in enumerate(batch_attmask):
            for j, element in enumerate(attmask):
                if element == 0:
                    ground_truth[i][j] = -100

        decoder_output = torch.swapaxes(decoder_output, 1, 2)
        self.loss = self.criterion(decoder_output, ground_truth)
        return self.loss

    def beam_search(self, img, beams=3, max_length=30):
        self.eval()

        def forward_prob(x: Tensor, encoder_feature: Tensor):
            x = torch.narrow(x, 1, 0, min(x.size(1), self.decoder.block_size))
            pos = torch.arange(
                x.size()[1], dtype=torch.long, device=x.device
            ).unsqueeze(0)
            x = self.decoder.transformer.wte(x) + self.decoder.transformer.wpe(pos)
            for block in self.decoder.transformer.h:
                x = block(x, encoder_feature)
            # Generator
            # 根據seq的最後一個字分類
            x = self.decoder.lm_head(self.decoder.transformer.ln_f(x[:, -1, :]))
            return x

        if img.dim() < 4:
            img = img.unsqueeze(0)
        encoder_feature = self.encoder.forward_features(img)
        encoder_feature = self.feature_resize(encoder_feature)
        cur_state = torch.tensor([BOS_TOKEN]).to(device).unsqueeze(1)
        ### Beam Search Start ###
        # get top k words
        next_probs = forward_prob(cur_state, encoder_feature)

        vocab_size = next_probs.shape[-1]
        # 選擇概率最高的beams個單詞作為初始候選序列

        # probs, pred id
        cur_probs, next_chars = next_probs.log_softmax(-1).topk(k=beams, axis=-1)
        cur_probs = cur_probs.reshape(beams)
        next_chars = next_chars.reshape(beams, 1)
        # gen first k beams
        cur_state = cur_state.repeat((beams, 1))  # 複製 beams 次
        cur_state = torch.cat((cur_state, next_chars), axis=1)

        ans_ids = []
        ans_probs = []
        for i in range(max_length - 1):
            # get top k beams for beam*beam candidates
            # print("current state: ", cur_state)
            next_probs = forward_prob(
                cur_state, encoder_feature.repeat((beams, 1, 1))
            ).log_softmax(-1)
            cur_probs = cur_probs.unsqueeze(-1) + next_probs
            cur_probs = cur_probs.flatten()  # (beams*vocab) 攤平成1D

            # length normalization
            # cur_probs / (len(cur_state[0]) + 1) -> nomalized
            _, idx = (cur_probs / (len(cur_state[0]) + 1)).topk(k=beams, dim=-1)
            cur_probs = cur_probs[idx]

            # get corresponding next char
            next_chars = torch.remainder(idx, vocab_size)
            next_chars = next_chars.unsqueeze(-1)
            # print("next char: ",next_chars)

            # get corresponding original beams
            top_candidates = (idx / vocab_size).long()  # 找回屬於哪個beam
            cur_state = cur_state[top_candidates]
            cur_state = torch.cat((cur_state, next_chars), dim=1)

            # concat next_char to beams
            to_rm_idx = set()
            for idx, ch in enumerate(next_chars):
                if i == (max_length - 2) or ch.item() == EOS_TOKEN:
                    ans_ids.append(cur_state[idx].cpu().tolist())
                    # print(cur_probs[idx].item()," / ",len(ans_ids[-1]))
                    ans_probs.append(cur_probs[idx].item() / len(ans_ids[-1]))
                    to_rm_idx.add(idx)
                    beams -= 1

            to_keep_idx = [i for i in range(len(cur_state)) if i not in to_rm_idx]
            if len(to_keep_idx) == 0:
                break
            cur_state = cur_state[to_keep_idx]
            cur_probs = cur_probs[to_keep_idx]

        max_idx = torch.argmax(torch.tensor(ans_probs)).item()

        # 把50256抽離
        ans_ids[max_idx] = [x for x in ans_ids[max_idx] if x != EOS_TOKEN]
        # print(ans_ids)
        return ans_ids[max_idx]

    def greedy_search(self, img, max_length=30):
        def forward_prob(x: Tensor, encoder_feature: Tensor):
            x = torch.narrow(x, 1, 0, min(x.size(1), self.decoder.block_size))
            pos = torch.arange(
                x.size()[1], dtype=torch.long, device=x.device
            ).unsqueeze(0)
            x = self.decoder.transformer.wte(x) + self.decoder.transformer.wpe(pos)
            for block in self.decoder.transformer.h:
                x = block(x, encoder_feature)
            # Generator
            # 根據seq的最後一個字分類
            x = self.decoder.lm_head(self.decoder.transformer.ln_f(x[:, -1, :]))
            return x

        self.eval()
        if img.dim() < 4:
            img = img.unsqueeze(0)
        device = img.device
        with torch.no_grad():
            encoder_feature = self.encoder.forward_features(img)
            encoder_feature = self.feature_resize(encoder_feature)

        cur_state = torch.tensor([EOS_TOKEN]).to(device).unsqueeze(1)
        for _ in range(max_length):
            with torch.no_grad():
                next_prob = forward_prob(cur_state, encoder_feature)

            next_word = next_prob.argmax(dim=-1).unsqueeze(0)
            if next_word.item() == EOS_TOKEN:
                break
            cur_state = torch.concat((cur_state, next_word), dim=-1)
        return cur_state[0, 1:].cpu().tolist()  # remove [BOS]


def visualize_attention(img, querys, keys, output_ids, img_name):
    tokenizer = BPETokenizer()
    img = img.squeeze(0).permute(1, 2, 0).cpu()
    img = (img - img.min()) / (img.max() - img.min())

    num_cols = 4  # 每row兩個
    num_plots = len(querys)
    num_rows = math.ceil(num_plots / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6))
    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    for i in range(len(querys)):
        ax = axes[i // num_cols, i % num_cols]
        title = tokenizer.decode([output_ids[i]]) # q = [b,tgt l,768] k = [b,src l ,768] -> q@k.trans -> [tgt l , src l] -> [1,16,16] -> 224,224
        att = querys[i] @ keys[i].permute(1, 0) * (1.0 / math.sqrt(keys[i].size(-1)))
        att = att[-1, 1:].view(1, 16, 16)
        attention_resized = F.interpolate(
            att.unsqueeze(0), size=img.shape[:2], mode="bilinear", align_corners=False
        )

        # plt.imshow(attention_resized, cmap="jet", alpha=0.5)  # 使用半透明的熱圖

        ax.imshow(img.cpu())
        ax.set_title(f"{title}")

        if i != 0:
            ax.imshow(attention_resized.squeeze().cpu().numpy(), cmap="jet", alpha=0.5)

    for i in range(num_plots, num_rows * num_cols):
        fig.delaxes(axes.flatten()[i])

    # plt.colorbar()
    plt.savefig(f"{img_name}.png")


def main():
    # args parameters
    cross_att_cfg_path = "model_adapter_ViT_4.pt"
    print("Load model: ", cross_att_cfg_path)
    cross_att_cfg = torch.load(cross_att_cfg_path)

    # Dataloader setting
    # 根據timm model config 去設定transform條件
    transform = create_transform(
        **resolve_data_config(
            {}, model="vit_large_patch14_clip_224.openai_ft_in12k_in1k"
        )
    )

    val_dir = "../hw3_data/p3_data/images"

    val_dataset = getDataset(
        img_dir=val_dir,
        transform=transform,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )

    # Model
    model = ImgCaptionModel().to(device)
    model.load_state_dict(cross_att_cfg, strict=False)

    # hook
    hook_q = []
    hook_k = []

    def fetch_q(module, input, output):
        query = output[1].detach()  # 確保與當前計算圖脫離
        hook_q.append(query.squeeze(0))

    def fetch_k(module, input, output):
        key = output[2].detach()  # 確保與當前計算圖脫離
        hook_k.append(key.squeeze(0))

    # validation part
    model.eval()
    for val_data in tqdm(val_loader):
        # 設定hook
        hook_q = []
        hook_k = []
        for block in model.decoder.transformer.h:
            block.crossattn.register_forward_hook(fetch_q)
            block.crossattn.register_forward_hook(fetch_k)

        ori_img, img, filename = val_data
        img = img.to(device)

        with torch.autocast(device_type="cuda"):
            output_ids = model.greedy_search(img)

        output_ids.insert(0, EOS_TOKEN)
        output_ids.insert(len(output_ids), EOS_TOKEN)
        tokenizer = BPETokenizer()
        print(tokenizer.decode(output_ids))
        
        # for q in hook_q:
        #     print(q.shape)
            
        querys = []
        keys = []
        querys = [hook_q[i] for i in range(1, len(hook_q)) if i % 11 == 0]
        keys = [hook_k[i] for i in range(1, len(hook_k)) if i % 11 == 0]

        visualize_attention(
            ori_img,
            querys[: len(output_ids)],
            keys[: len(output_ids)],
            output_ids,
            filename,
        )


if __name__ == "__main__":
    main()
