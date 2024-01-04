import os
import torch
import imageio
import numpy as np
from tqdm import tqdm
from models.nerf import *
from utils import load_ckpt
from dataset import KlevrDataset
from collections import defaultdict
from argparse import ArgumentParser
from models.rendering import render_rays



torch.backends.cudnn.benchmark = True


def get_opts():
    parser = ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        help="root directory of dataset",
    )

    parser.add_argument(
        "--dir_name",
        type=str,
        help="output file dir",
    )

    return parser.parse_args()


@torch.no_grad()
def batched_inference(
    models, embeddings, rays, N_samples, N_importance, use_disp, chunk
):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    chunk = 1024 * 32
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = render_rays(
            models,
            embeddings,
            rays[i : i + chunk],
            N_samples,
            use_disp,
            0,
            0,
            N_importance,
            chunk,
        )

        for k, v in rendered_ray_chunks.items():
            results[k] += [v]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results


if __name__ == "__main__":
    args = get_opts()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = KlevrDataset(split="test", root_dir=args.root_dir)
    w = dataset.meta["metadata"]["width"]
    h = dataset.meta["metadata"]["width"]
    model_path = "model_ckpt/hw4.ckpt"
    dir_name = args.dir_name

    embedding_xyz = Embedding(3, 10)
    embedding_dir = Embedding(3, 4)
    nerf_fine = NeRF().to(device)
    nerf_course = NeRF().to(device)
    load_ckpt(nerf_fine, model_path, model_name="nerf_fine")
    load_ckpt(nerf_course, model_path, model_name="nerf_course")
    nerf_fine.cuda().eval()

    models = [nerf_course, nerf_fine]
    embeddings = [embedding_xyz, embedding_dir]

    imgs = []
    psnrs = []

    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        rays = sample["rays"].cuda()
        results = batched_inference(
            models,
            embeddings,
            rays,
            64,
            128,
            False,
            32 * 1024 * 4,
        )

        img_pred = results["rgb_fine"].view(h, w, 3).cpu().numpy()

        img_pred_ = (img_pred * 255).astype(np.uint8)
        imgs += [img_pred_]
        imageio.imwrite(
            os.path.join(dir_name, f"{dataset.meta['split_ids']['test'][i]:05d}.png"),
            img_pred_,
        )
