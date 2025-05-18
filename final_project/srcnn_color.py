import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import load_dataset, preprocess_ycbcr, calc_psnr

# SRCNN 模型 (對 Y 通道)
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# 訓練模型
def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    for epoch in range(5):  # 可自行調整
        running_loss = 0.0
        for lr, hr in dataloader:
            lr, hr = lr.to(device), hr.to(device)
            optimizer.zero_grad()
            outputs = model(lr)
            loss = criterion(outputs, hr)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"[Epoch {epoch+1}] Loss: {running_loss/len(dataloader):.6f}")

# 主程序
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SRCNN().to(device)

    train_loader = load_dataset('data/train', batch_size=16)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_model(model, train_loader, criterion, optimizer, device)

    # 儲存模型
    os.makedirs('weights', exist_ok=True)
    torch.save(model.state_dict(), 'weights/srcnn_trained.pth')
    print("✅ 訓練完成，模型已儲存。")
