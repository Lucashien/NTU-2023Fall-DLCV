import json
import timm
import math
import torch
import os, sys
import collections
import torch.nn as nn
from PIL import Image
from tqdm.auto import tqdm
from torch import Tensor, nn
import torch.nn.functional as F
from tokenizer import BPETokenizer
from timm.data import resolve_data_config
from torch.utils.data import DataLoader, Dataset
from timm.data.transforms_factory import create_transform

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {device}")
PAD_TOKEN = 50256
EOS_TOKEN = 50256


def pad_sequences(sequences, pad_token_id=0):
    max_length = max([len(seq) for seq in sequences])
    padded_sequences = [
        seq + [pad_token_id] * (max_length - len(seq)) for seq in sequences
    ]
    return padded_sequences


class getDataset(Dataset):
    def __init__(self, img_dir, transform):
        super().__init__()
        print(f"Loading img from {img_dir}")

        self.img_dir = img_dir
        if img_dir.endswith("/"):
            self.img_dir = self.img_dir
        else:
            self.img_dir = img_dir + "/"

        self.image_files = []
        print(self.img_dir)
        for _, _, filenames in os.walk(img_dir):
            for filename in filenames:
                self.image_files.append(filename)

        self.transform = transform

    def __getitem__(self, index):
        imgname = self.image_files[index]
        img = Image.open(self.img_dir + imgname).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {
            "image": img,
            "filename": os.path.splitext(imgname)[0],
        }

    def __len__(self):
        return len(self.image_files)


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
        x = x + self.crossattn(x, encoder_output)
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


class ImgCaptionModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Tokenizer
        self.tokenizer = BPETokenizer("encoder.json","vocab.bpe")

        # Encoder (pretrained ViT model)
        self.encoder = timm.create_model(
            "vit_large_patch14_clip_224.openai_ft_in12k_in1k",
            pretrained=True,
            num_classes=0,
        ).to(device)
        self.feature_resize = nn.Linear(1024, 768)

        # Decoder
        self.cfg = cfg
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
        if img.dim() < 4:
            img = img.unsqueeze(0)

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

        encoder_feature = self.encoder.forward_features(img)
        encoder_feature = self.feature_resize(encoder_feature)
        id_list = []
        prpb_list = []

        # ----------/Beam Searching/-----------#
        cur_state = torch.tensor([EOS_TOKEN]).to(device).unsqueeze(1)
        n_probs = forward_prob(cur_state, encoder_feature)

        vocab_size = n_probs.shape[-1]

        # 選擇概率最高的beams個單詞作為初始候選序列
        # probs, pred id
        cur_probs, n_chars = n_probs.log_softmax(-1).topk(k=beams, axis=-1)
        cur_probs = cur_probs.reshape(beams)
        n_chars = n_chars.reshape(beams, 1)
        # gen first k beams
        cur_state = cur_state.repeat((beams, 1))  # 複製 beams 次
        cur_state = torch.cat((cur_state, n_chars), axis=1)

        for i in range(max_length - 1):
            rm_set = set()
            # to get top k
            n_probs = forward_prob(
                cur_state, encoder_feature.repeat((beams, 1, 1))
            ).log_softmax(-1)

            cur_probs = (
                cur_probs.unsqueeze(-1) + n_probs
            ).flatten()  # (beams*vocab) 攤平成1D

            # length normalization
            _, idx = (cur_probs / (len(cur_state[0]) + 1)).topk(k=beams, dim=-1)
            cur_probs = cur_probs[idx]

            # to generate next char
            n_chars = (torch.remainder(idx, vocab_size)).unsqueeze(-1)

            # 找回屬於哪個beam
            top_k = (idx / vocab_size).long()
            cur_state = torch.cat((cur_state[top_k], n_chars), dim=1)

            # concat next_char to beams
            for idx, char in enumerate(n_chars):
                if char.item() == EOS_TOKEN or cur_state.size(1) == max_length:
                    id_list.append(cur_state[idx].cpu().tolist())
                    prpb_list.append(cur_probs[idx].item() / len(id_list[-1]))
                    rm_set.add(idx)
                    beams -= 1

            to_keep_idx = [i for i in range(len(cur_state)) if i not in rm_set]
            if len(to_keep_idx) == 0:
                break
            cur_state = cur_state[to_keep_idx]
            cur_probs = cur_probs[to_keep_idx]

        max_idx = torch.argmax(torch.tensor(prpb_list)).item()

        # 把50256抽離
        id_list[max_idx] = [x for x in id_list[max_idx] if x != EOS_TOKEN]
        return id_list[max_idx]


def save_json(json_path, filename, pred_caption):
    pred_caption_list = []
    pred_caption_list.append(pred_caption)
    new_data = dict(zip(filename, pred_caption_list))

    try:
        with open(f"{json_path}", "r") as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        with open(f"{json_path}", "w") as file:
            data = {}

    data.update(new_data)
    with open(f"{json_path}", "w") as f:
        json.dump(data, f)


def main():
    # args parameters
    tokenizer = BPETokenizer("encoder.json","vocab.bpe")
    val_dir = sys.argv[1]
    output_json = sys.argv[2]
    cfg_path = sys.argv[3]
    cfg = Config(cfg_path)

    cross_att_cfg_path = "model/model_P2_best.pt"
    cross_att_cfg = torch.load(cross_att_cfg_path)

    print("Load model: ", cross_att_cfg_path)
    print("Write into ", output_json)

    # Dataloader setting
    # 根據timm model config 去設定transform條件
    transform = create_transform(
        **resolve_data_config(
            {}, model="vit_large_patch14_clip_224.openai_ft_in12k_in1k"
        )
    )

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
    model = ImgCaptionModel(cfg).to(device)
    model.load_state_dict(cross_att_cfg, strict=False)

    # validation part
    model.eval()
    for val_data in tqdm(val_loader):
        val_data["image"] = val_data["image"].to(device)
        output_ids = model.beam_search(val_data["image"], beams=5)
        sentence = tokenizer.decode(output_ids)
        save_json(output_json, val_data["filename"], sentence)


if __name__ == "__main__":
    main()
