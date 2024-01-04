import argparse
import csv
import json
import os, sys
import pathlib
import torch
from torch import Tensor, nn
from PIL import Image
from tqdm.auto import tqdm
import torch.nn.functional as F
from tokenizer import BPETokenizer
import timm
import torchvision.transforms as trns
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import math
import collections
import torch
from torch import nn, Tensor

# logging.basicConfig(
#     filename="my_log.log",
#     level=logging.INFO,
#     format="%(message)s",
# )

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {device}")
PAD_TOKEN = 50256
UNK_TOKEN = 1
BOS_TOKEN = 50256
EOS_TOKEN = 50256


def pad_sequences(sequences, pad_token_id=0):
    max_length = max([len(seq) for seq in sequences])
    padded_sequences = [
        seq + [pad_token_id] * (max_length - len(seq)) for seq in sequences
    ]
    return padded_sequences


class getDataset(Dataset):
    def __init__(self, img_dir, json_file, transform):
        super().__init__()
        print(f"Loading img from {img_dir}")
        print(f"Loading json from {json_file}")
        with open(json_file, "r") as file:
            info = json.load(file)
        self.tokenizer = BPETokenizer()
        self.img_dir = img_dir
        self.transform = transform
        self.data = []
        self.id2img = {}

        # notation
        for data in info["annotations"]:
            entry = {"caption": data["caption"], "image_id": data["image_id"]}
            self.data.append(entry)

        # img file
        for data in info["images"]:
            self.id2img[data["id"]] = data["file_name"]

    def __getitem__(self, index):
        info = self.data[index]  # {"caption":xxx , "image_id":xxx}
        imgname = self.id2img[info["image_id"]]
        img = Image.open(self.img_dir + "/" + imgname).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {
            "image": img,
            "caption": info["caption"],
            "filename": os.path.splitext(imgname)[0],
        }

    def __len__(self):
        return len(self.data)

    # retrun 一整個batch的dict
    def collate_fn(self, samples):
        captions2id = list()
        filenames = list()
        images = list()
        Start_token = 50256

        for sample in samples:
            id = self.tokenizer.encode(sample["caption"])
            if id[0] != Start_token:
                id.insert(0, Start_token)
            if id[-1] != Start_token:
                id.insert(len(id), Start_token)
            images.append(sample["image"])
            captions2id.append(id)
            filenames.append(sample["filename"])

        pad_captions2id = pad_sequences(captions2id, -1)
        attention_masks = [[float(i != -1) for i in seq] for seq in pad_captions2id]

        pad_captions2id = [
            [PAD_TOKEN if x == -1 else x for x in seq] for seq in pad_captions2id
        ]

        captions = torch.tensor(pad_captions2id)
        attention_mask_tensors = torch.tensor(attention_masks)
        images = torch.stack(images, dim=0)
        return {
            "images": images,
            "captions": captions,
            "filenames": filenames,
            "attmask": attention_mask_tensors,
        }


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

    def forward(self, x, encoder_output, prefix_emb=None):
        attn_output = self.attn(self.ln_1(x))

        if prefix_emb is not None:
            prefix_k, prefix_v = (
                prefix_emb[:, self.layer_idx, 0, :],
                prefix_emb[:, self.layer_idx, 1, :],
            )
            attn_output += self.attn.add_prefix(prefix_k, prefix_v)
        x = x + attn_output

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

        num_virtual_tokens = 30
        num_layers = self.cfg.n_layer
        token_dim = 1024
        # (30, 49152=24*2*1024) -> Embedding(30, 49152)
        """
            num_virtual_tokens: 虛擬token
            num_layers: transformer decoder layers = 12
            *2: key, value
        """
        self.prefix = nn.Embedding(num_virtual_tokens, num_layers * 2 * token_dim)
        self.prompt_tokens = torch.arange(num_virtual_tokens).long()

    # x is id that transform by captions
    def forward(self, x: Tensor, encoder_feature: Tensor):
        # narrow (dim,start,len)
        x = torch.narrow(x, 1, 0, min(x.size(1), self.block_size))
        pos = torch.arange(x.size()[1], dtype=torch.long, device=x.device).unsqueeze(0)

        prefix_emb = self.prefix(
            self.prompt_tokens.to(device)
        )  # 將 prompt_tokens 轉移到適當的設備上
        prefix_emb = prefix_emb.view(
            -1, self.cfg.n_layer, 2, self.cfg.n_embd
        )  # 重構形狀以匹配 Layer 和 Head 維度

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
            "vit_large_patch14_clip_224", pretrained=True, num_classes=0
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

        # # test block
        # _, output_id = torch.max(decoder_output[0], dim=-1)
        # self.test = self.tokenizer.decode(output_id.tolist())
        # print(self.test)

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


def norm_long(x):
    x /= x.norm(dim=-1, keepdim=True)
    return x.long()


def save_json(json_path, filename, pred_caption):
    pred_caption_list = []
    pred_caption_list.append(pred_caption)
    new_data = dict(zip(filename, pred_caption_list))
    # json_file = sys.argv[2]
    # 嘗試讀取現有數據
    try:
        with open(f"{json_path}", "r") as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        with open(f"{json_path}", "w") as file:
            data = {}

    data.update(new_data)
    with open(f"{json_path}", "w") as f:
        json.dump(data, f)


def main(file_path, mode):
    # args parameters
    tokenizer = BPETokenizer()
    cfg_path = "../hw3_data/p2_data/decoder_model.bin"
    cross_att_cfg_path = file_path
    print("Load model: ", cross_att_cfg_path)
    cfg = Config(cfg_path)
    cross_att_cfg = torch.load(cross_att_cfg_path)

    json_path = f"{file_path.split('.')[0]}_{mode}.json"
    print("Write into ", json_path)

    # Dataloader setting
    # 根據timm model config 去設定transform條件
    transform = create_transform(
        **resolve_data_config({}, model="vit_large_patch14_clip_224")
    )

    val_dir = "../hw3_data/p2_data/images/val"
    val_json = "../hw3_data/p2_data/val.json"

    val_dataset = getDataset(
        img_dir=val_dir,
        json_file=val_json,
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
    # validation part
    model.eval()
    for val_data in tqdm(val_loader):
        val_data["image"] = val_data["image"].to(device)
        with torch.autocast(device_type="cuda"):
            if mode == "g":
                output_ids = model.greedy_search(val_data["image"])
            elif mode == "b":
                output_ids = model.beam_search(val_data["image"], beams=3)

        sentence = tokenizer.decode(output_ids)
        save_json(json_path, val_data["filename"], sentence)


if __name__ == "__main__":
    folder_path = os.getcwd()
    mode = sys.argv[1]

    for filename in os.listdir(folder_path):
        if (
            filename.endswith("_0.pt")
            or filename.endswith("_1.pt")
            or filename.endswith("_2.pt")
            or filename.endswith("_3.pt")
            or filename.endswith("_4.pt")
            or filename.endswith("_5.pt")
            or filename.endswith("_6.pt")
            or filename.endswith("_7.pt")
            or filename.endswith("_8.pt")
            or filename.endswith("_9.pt")
        ) and "prefix" in filename:
            file_path = os.path.join(folder_path, filename)
            main(file_path, mode)
