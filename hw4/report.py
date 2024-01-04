import torch
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import imageio
from argparse import ArgumentParser

from models.rendering import render_rays
from models.nerf import *

from utils import load_ckpt
import metrics

from dataset import KlevrDataset
import matplotlib.pyplot as plt
from utils import *

torch.backends.cudnn.benchmark = True


def get_opts():
    parser = ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/home/ubuntu/data/nerf_example_data/nerf_synthetic/lego",
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
            test_time=True,
        )

        for k, v in rendered_ray_chunks.items():
            results[k] += [v]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results


if __name__ == "__main__":
    args = get_opts()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = KlevrDataset(split="val", root_dir=args.root_dir)
    w = dataset.meta["metadata"]["width"]
    h = dataset.meta["metadata"]["width"]
    model_path = "logs/my_experiment/version_4/checkpoints/ckpts/exp/epoch=12.ckpt"
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
    fig, axes = plt.subplots(5, 4, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.4)
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
            os.path.join(dir_name, f"{dataset.meta['split_ids']['val'][i]:05d}.png"),
            img_pred_,
        )

        if "rgbs" in sample:
            rgbs = sample["rgbs"]
            img_gt = rgbs.view(h, w, 3)
            psnrs += [metrics.psnr(img_gt, img_pred).item()]

        row_idx = i // 4
        col_idx = i % 4
        ax = axes[row_idx,col_idx]       
        ax.imshow(visualize_depth(results["depth_fine"].view(w, h)).permute(1, 2, 0))
        ax.set_title( f"{dataset.meta['split_ids']['val'][i]:05d}.png")

    imageio.mimsave(
        os.path.join(dir_name, f"all_test.gif"),
        imgs,
        fps=30,
    )

    if psnrs:
        mean_psnr = np.mean(psnrs)
        print(f"Mean PSNR : {mean_psnr:.2f}")

    plt.savefig("depth.png")
