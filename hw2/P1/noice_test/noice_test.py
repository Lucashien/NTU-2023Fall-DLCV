import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms

IMG_SIZE = 28
BATCH_SIZE = 128

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

img_transform = [
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # Scales data into [0,1]
    transforms.Lambda(lambda x: x.to(device)),
    transforms.Lambda(lambda t: (t * 2) - 1),  # Scale between [-1, 1]
]
img_transform = transforms.Compose(img_transform)

# 載入dataloader以及顯示部分圖片
pic_path = "./P1/noice_test"
num_samples = 8

data = torchvision.datasets.ImageFolder(root=pic_path, transform=img_transform)

plt.figure(figsize=(10, 10))

for i, img in enumerate(data):
    plt.subplot(num_samples // 4 + 1, 4, i + 1)
    plt.imshow(torch.permute(img[0].cpu(), (1, 2, 0)))  # (c, h, w) -> (h, w, c)

plt.savefig(f"P1/noice_test/pic/intput.png")


def forward_diffuse_process(x_0, t):
    """
    return diffused image with given x_0 and timestep
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t]
    sqrt_oneminus_alphas_cumprod_t = sqrt_oneminus_alphas_cumprod[t]

    return sqrt_alphas_cumprod_t * x_0 + sqrt_oneminus_alphas_cumprod_t * noise, noise


def linear_schedule(timesteps=500, start=0.0001, end=0.02):
    """
    return a tensor of a linear schedule
    """
    return torch.linspace(start, end, timesteps)


# precalculations
T = 200
betas = linear_schedule(timesteps=T)
alphas = 1 - betas

alphas_cumprod = torch.cumprod(alphas, dim=0)  # 壘乘
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_oneminus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

import numpy as np

# Simulate forward diffusion
image = next(iter(data))[0]

plt.figure(figsize=(15, 15))
plt.axis("off")
num_images = 10
stepsize = int(T / num_images)


def show_tensor_image(image):
    reverse_transforms = transforms.Compose(
        [
            transforms.Lambda(lambda t: (t + 1) / 2),
            transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
            transforms.Lambda(lambda t: t * 255.0),
            transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ]
    )

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image))


for idx in range(0, T, stepsize):
    t = idx
    plt.subplot(1, num_images + 1, (idx // stepsize) + 1)
    image, noise = forward_diffuse_process(image, t)
    show_tensor_image(image)

plt.savefig("P1/noice_test/pic/output.png")
