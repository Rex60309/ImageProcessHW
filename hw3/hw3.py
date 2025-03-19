import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def apply_average_filter(image, kernel_size):
    """
    對圖像應用平均濾波器

    Parameters:
    image (numpy.ndarray): 輸入圖像
    kernel_size (int): 濾波器的大小 (n x n)

    Returns:
    numpy.ndarray: 處理後的圖像
    """
    # 建立一個全為1的核心
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    # 應用濾波器
    return cv2.filter2D(image, -1, kernel)

def apply_median_filter(image, kernel_size):
    """
    對圖像應用中值濾波器

    Parameters:
    image (numpy.ndarray): 輸入圖像
    kernel_size (int): 濾波器的大小 (n x n)

    Returns:
    numpy.ndarray: 處理後的圖像
    """
    return cv2.medianBlur(image, kernel_size)

def apply_unsharp_masking(image, kernel_size, amount=1.0):
    """
    對圖像應用非銳化遮罩技術

    Parameters:
    image (numpy.ndarray): 輸入圖像
    kernel_size (int): 用於模糊的濾波器大小
    amount (float): 銳化的強度

    Returns:
    numpy.ndarray: 處理後的圖像
    """
    # 步驟1: 將原圖高斯模糊
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    # 步驟2: 計算遮罩（原圖 - 模糊圖）
    mask = cv2.subtract(image, blurred)

    # 步驟3: 將遮罩加到原圖
    sharpened = cv2.addWeighted(image, 1.0 + amount, mask, amount, 0)

    return sharpened

def main():
    image_path = input("Enter image path: ").strip()
    if not Path(image_path).exists():
        print(f"Error: can't find picture from '{image_path}'")
        return

    original_image = cv2.imread(image_path)

    # 如果圖像是彩色的，轉換為BGR到RGB
    if len(original_image.shape) == 3:
        original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    else:
        original_image_rgb = original_image

    # 指定核心大小
    kernel_size = 5  # 可以改變這個值來測試不同大小的濾波器效果

    # 應用平均濾波器
    avg_filtered = apply_average_filter(original_image_rgb, kernel_size)

    # 應用中值濾波器
    median_filtered = apply_median_filter(original_image_rgb, kernel_size)

    # 應用非銳化遮罩
    unsharp_masked = apply_unsharp_masking(original_image_rgb, kernel_size, amount=1.5)

    # 顯示結果
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.title('Original Image')
    plt.imshow(original_image_rgb)
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title(f'{kernel_size}x{kernel_size} 平均濾波器')
    plt.imshow(avg_filtered)
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title(f'{kernel_size}x{kernel_size} 中值濾波器')
    plt.imshow(median_filtered)
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title('非銳化遮罩')
    plt.imshow(unsharp_masked)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('image_processing_results.png')
    plt.show()

    if len(original_image.shape) == 3:
        # 將處理後的圖像從RGB轉換回BGR以便使用cv2.imwrite保存
        avg_filtered_bgr = cv2.cvtColor(avg_filtered, cv2.COLOR_RGB2BGR)
        median_filtered_bgr = cv2.cvtColor(median_filtered, cv2.COLOR_RGB2BGR)
        unsharp_masked_bgr = cv2.cvtColor(unsharp_masked, cv2.COLOR_RGB2BGR)

        cv2.imwrite('average_filtered.jpg', avg_filtered_bgr)
        cv2.imwrite('median_filtered.jpg', median_filtered_bgr)
        cv2.imwrite('unsharp_masked.jpg', unsharp_masked_bgr)
    else:
        cv2.imwrite('average_filtered.jpg', avg_filtered)
        cv2.imwrite('median_filtered.jpg', median_filtered)
        cv2.imwrite('unsharp_masked.jpg', unsharp_masked)

    print("Pictures saved complete!")

if __name__ == "__main__":
    main()