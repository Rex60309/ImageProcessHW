from PIL import Image
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import CenterCrop
import numpy as np

# 專門處理 Y 通道
# utils.py 中的 preprocess_ycbcr：
def preprocess_ycbcr(img, upscale=2, crop_size=128):
    img = img.convert('YCbCr')
    y, _, _ = img.split()
    y = transforms.CenterCrop(crop_size)(y)

    lr_y = y.resize((crop_size // upscale, crop_size // upscale), Image.BICUBIC)
    lr_y = lr_y.resize((crop_size, crop_size), Image.BICUBIC)

    transform = transforms.ToTensor()  # Output shape: [1, H, W]
    return transform(lr_y), transform(y)  # ✅ OK


# 自訂資料集
class SRDataset(Dataset):
    def __init__(self, folder_path):
        self.image_files = [os.path.join(folder_path, x) for x in os.listdir(folder_path) if x.endswith(('.jpg', '.png'))]

    def __getitem__(self, index):
        img = Image.open(self.image_files[index])
        lr_y, hr_y = preprocess_ycbcr(img)
        return lr_y, hr_y  # ✅ 不要再用 unsqueeze(0)

    def __len__(self):
        return len(self.image_files)

# 載入資料集
def load_dataset(folder, batch_size=16):
    dataset = SRDataset(folder)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 計算 PSNR
def calc_psnr(sr, hr):
    mse = torch.mean((sr - hr) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
