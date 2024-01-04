from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import torchvision.transforms as trns
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import os
from PIL import Image
from UNet import UNet
from PIL import Image, ImageDraw, ImageFont


def beta_scheduler(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    return betas


class DDIM:
    def __init__(self, nn_model, timesteps=1000, beta_schedule=beta_scheduler()):
        self.model = nn_model
        self.timesteps = timesteps
        self.betas = beta_schedule

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # below: log calculation clipped because the posterior variance is 0 at the beginning
        # of the diffusion chain
        # self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min =1e-20))
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )

        self.posterior_mean_coef1 = (
            self.betas
            * torch.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

    # get the param of given timestep t
    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out

    # use ddim to sample
    @torch.no_grad()
    def sample(
        self,
        batch_size=10,
        ddim_timesteps=50,
        ddim_eta=0.0,
        clip_denoised=True,
    ):
        c = self.timesteps // ddim_timesteps
        ddim_timestep_seq = np.asarray(list(range(0, self.timesteps, c)))

        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        ddim_timestep_seq = ddim_timestep_seq + 1
        # previous sequence
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])

        device = next(self.model.parameters()).device
        # start from pure noise (for each example in the batch)
        filenames = [f"{i:02d}.pt" for i in range(0, batch_size)]
        # 讀取所有的 .pt 文件並將它們存儲到一個列表中
        tensors = [
            torch.load(os.path.join("hw2_data/face/noise", filename))
            for filename in filenames
        ]

        # 串連所有的張量
        # 想要沿著第0維串連張量，所以dim=0
        sample_img = torch.cat(tensors, dim=0)

        for i in tqdm(
            reversed(range(0, ddim_timesteps)),
            desc="Sampling loop time step",
            total=ddim_timesteps,
        ):
            t = torch.full(
                (batch_size,), ddim_timestep_seq[i], device=device, dtype=torch.long
            )
            prev_t = torch.full(
                (batch_size,),
                ddim_timestep_prev_seq[i],
                device=device,
                dtype=torch.long,
            )

            # 1. get current and previous alpha_cumprod
            alpha_cumprod_t = self._extract(self.alphas_cumprod, t, sample_img.shape)
            alpha_cumprod_t_prev = self._extract(
                self.alphas_cumprod, prev_t, sample_img.shape
            )

            # 2. predict noise using model
            pred_noise = self.model(sample_img, t)

            # 3. get the predicted x_0
            pred_x0 = (
                sample_img - torch.sqrt((1.0 - alpha_cumprod_t)) * pred_noise
            ) / torch.sqrt(alpha_cumprod_t)

            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, min=-1.0, max=1.0)

            # 4. compute variance: "sigma_t(η)" -> see formula (16)
            # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            sigmas_t = ddim_eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev)
                / (1 - alpha_cumprod_t)
                * (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
            )

            # 5. compute "direction pointing to x_t" of formula (12)
            pred_dir_xt = (
                torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t**2) * pred_noise
            )

            # 6. compute x_{t-1} of formula (12)
            x_prev = (
                torch.sqrt(alpha_cumprod_t_prev) * pred_x0
                + pred_dir_xt
                + sigmas_t * torch.randn_like(sample_img)
            )

            sample_img = x_prev

        return sample_img.cpu()


def output_img(img_num=10, eta=0):
    # hardcoding these here
    n_T = 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = "P2/output/"
    UNet_pt_dir = "hw2_data/face/UNet.pt"
    unet_model = UNet()
    unet_model.load_state_dict(torch.load(UNet_pt_dir))

    ddim = DDIM(
        nn_model=unet_model.to(device),
        timesteps=n_T,
    )

    with torch.no_grad():
        x_gen = ddim.sample(batch_size=img_num, ddim_eta=eta)
        for i in range(len(x_gen)):
            img = x_gen[i]
            min_val = torch.min(img)
            max_val = torch.max(img)

            # Min-Max Normalization
            normalized_x_gen = (img - min_val) / (max_val - min_val)
            save_image(normalized_x_gen, save_dir + f"{i:02d}.png")


def Compare_mse():
    img_dir = "P2/output/"
    GT_dir = "hw2_data/face/GT/"
    img = [f"{img_dir}{i:02d}.png" for i in range(10)]
    GT = [f"{GT_dir}{i:02d}.png" for i in range(10)]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    compare_img_list = torch.empty(0, dtype=torch.float32)
    for i, (generated_path, ground_truth_path) in enumerate(zip(img, GT)):
        img = transform(Image.open(generated_path))
        GT = transform(Image.open(ground_truth_path))
        compare_img = torch.cat((img, GT), dim=2)
        compare_img_list = torch.cat((compare_img_list, compare_img), dim=1)

        # convert to 0~255
        img = normal_to_255(img)
        GT = normal_to_255(GT)

        # 確保大小一樣
        assert img.shape == GT.shape

        # Calc MSE
        mse = torch.nn.functional.mse_loss(img, GT)
        print(f"MSE for image pair {i}: {mse.item():5f}")

    save_dir = "P2/output/"
    save_image(compare_img_list, save_dir + f"compare.png")


def normal_to_255(tensor):
    for i, x in enumerate(tensor):
        new_values = []
        for row in x:
            new_row = []
            for value in row:
                new_value = int(value * 255)
                new_row.append(new_value)
            new_values.append(new_row)
        tensor[i] = torch.tensor(new_values)
    return tensor


def eta_compare():
    img_dir = "P2/output/"

    imgs_grid = torch.empty(0, dtype=torch.float32)

    for eta in np.arange(0, 1.25, 0.25):
        output_img(img_num=4, eta=eta)
        imgs = [f"{img_dir}{i:02d}.png" for i in range(4)]
        imgs_row = torch.empty(0, dtype=torch.float32)
        for img in imgs:
            img = transforms.ToTensor()(Image.open(img))
            imgs_row = torch.cat((imgs_row, img), dim=2)

        # 添加一列空白像素到左侧
        blank_column = torch.ones(3, 256, 256) * 255
        imgs_row = torch.cat((blank_column, imgs_row), dim=2)
        pil_image = transforms.ToPILImage()(imgs_row)  # Tensor to PIL
        draw = ImageDraw.Draw(pil_image)
        draw.text(
            xy=(blank_column.shape[0] // 3, blank_column.shape[1] // 2),
            text=f"eta = {eta}: ",
            fill=(255, 255, 255),
        )
        imgs_row = transforms.ToTensor()(pil_image)
        imgs_grid = torch.cat((imgs_grid, imgs_row), dim=1)

    save_image(imgs_grid, img_dir + "grid.png")


if __name__ == "__main__":
    # output_img()
    # eta_compare()
    Compare_mse()
