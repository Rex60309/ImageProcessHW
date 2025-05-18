import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from srcnn_color import SRCNN
from utils import preprocess_ycbcr
import numpy as np
import os

# 設定裝置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 載入模型
model = SRCNN().to(device)
model.load_state_dict(torch.load('weights/srcnn_trained.pth', map_location=device))
model.eval()

# 預處理與還原工具
to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

# 載入圖片
img = Image.open('data/test/img_001_SRF_2_LR.png')  # 改成你的圖片路徑
img = img.convert('YCbCr')
y, cb, cr = img.split()

# 保留原始 Cb/Cr
cb = cb.resize(y.size, Image.BICUBIC)
cr = cr.resize(y.size, Image.BICUBIC)

# 製作低解析 → 放大成 bicubic LR
crop_size = 128
upscale = 2
y_cropped = transforms.CenterCrop(crop_size)(y)
cb_cropped = transforms.CenterCrop(crop_size)(cb)
cr_cropped = transforms.CenterCrop(crop_size)(cr)

lr_y = y_cropped.resize((crop_size // upscale, crop_size // upscale), Image.BICUBIC)
lr_y_up = lr_y.resize((crop_size, crop_size), Image.BICUBIC)  # 再上採樣

# 預處理 Y 通道
input_tensor = to_tensor(lr_y_up).unsqueeze(0).to(device)

# 推論
with torch.no_grad():
    out_y = model(input_tensor).clamp(0.0, 1.0)

# 回到 PIL Image
out_y_img = to_pil(out_y.squeeze(0).cpu())

# 還原為彩色圖（將 Y 預測 + 原 CbCr 合併）
final_img = Image.merge("YCbCr", [out_y_img, cb_cropped, cr_cropped]).convert("RGB")

# 顯示比較
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(Image.merge("YCbCr", [lr_y_up, cb_cropped, cr_cropped]).convert("RGB"))
plt.title("Bicubic Upsample")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(final_img)
plt.title("SRCNN Color Output")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(Image.merge("YCbCr", [y_cropped, cb_cropped, cr_cropped]).convert("RGB"))
plt.title("Ground Truth")
plt.axis('off')

plt.tight_layout()
plt.show()

# 可選：儲存圖片
os.makedirs("results", exist_ok=True)
final_img.save("data/results/srcnn_result_rgb.jpg")
print("✅ 已完成彩色圖像重建，已儲存至 data/results/srcnn_result_rgb.jpg")
