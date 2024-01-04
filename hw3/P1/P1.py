import argparse
import csv
import json
import os
import pathlib

import clip
import torch
from PIL import Image
from tqdm.auto import tqdm


def loading_data():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=pathlib.Path)
    parser.add_argument("--json", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    with args.json.open("r") as jf:
        labels = [v for n, v in json.load(jf).items()]

    imgs = [img for img in args.file.glob("*")]
    output_csv = args.output
    return imgs, labels, output_csv


def write_csv(csv_file, img_name, pred):
    with csv_file.open(mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("filename", "label"))
        for data in zip(img_name, pred):
            writer.writerow(data)


def print_top5(similarity, labels):
    top5_value, top5_name = similarity[0].topk(5)
    top5_mapped = [labels[idx] for idx in top5_name]
    for i, name in enumerate(top5_mapped):
        print(f"{name}: {100*top5_value[i]}%")


def main():
    # args parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_names, labels, output_csv = loading_data()
    model, preprocess = clip.load("ViT-L/14", device=device)
    correct = 0

    img_n_list = []
    pred_list = []

    for name in tqdm(img_names):
        img_input = Image.open(name).convert("RGB")
        img_input = preprocess(img_input).unsqueeze(0).to(device)
        text_inputs = torch.cat(
            [clip.tokenize(f"a photo of a {label}.") for label in labels]
        ).to(device)
        with torch.no_grad():
            image_features = model.encode_image(img_input)  # (1, 768)
            text_features = model.encode_text(text_inputs)  # (50, 768)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # print(image_features @ text_features.T)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        pred = similarity[0].argmax()
        img_n_list.append(name.name)
        pred_list.append(pred.item())
        # print_top5(similarity,labels)

    for name, label in zip(img_n_list, pred_list):
        correct += int(int(name.split("_")[0]) == label)

    print(f"accuracy: {correct / len(img_n_list)}")
    write_csv(output_csv, img_n_list, pred_list)


if __name__ == "__main__":
    main()
