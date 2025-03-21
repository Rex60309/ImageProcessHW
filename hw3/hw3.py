import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# https://www.geeksforgeeks.org/spatial-filters-averaging-filter-and-median-filter-in-image-processing/

# average filter
def average_filter(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)

    return cv2.filter2D(image, -1, kernel)

# median filter
def median_filter(image, kernel_size):
    return cv2.medianBlur(image, kernel_size)

# unsharp masking
def unsharp_masking(image, kernel_size, amount=1.0):
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
        print(f"Error: Can't load image from '{image_path}'")
        return

    original_image = cv2.imread(image_path)

    # BGR To RGB
    if len(original_image.shape) == 3:
        original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    else:
        original_image_rgb = original_image

    # Kernel size
    kernel_size = int(input("Enter kernel size (odd integer): "))

    # average filter
    avg_filtered = average_filter(original_image_rgb, kernel_size)

    # median filter
    median_filtered = median_filter(original_image_rgb, kernel_size)

    # unsharp masking
    unsharp_masked = unsharp_masking(original_image_rgb, kernel_size, amount=1.5)

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.title('Original Image')
    plt.imshow(original_image_rgb)
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title(f'{kernel_size}x{kernel_size} Average Filter')
    plt.imshow(avg_filtered)
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title(f'{kernel_size}x{kernel_size} Median Filter')
    plt.imshow(median_filtered)
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title('Unsharp Masking')
    plt.imshow(unsharp_masked)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('image_processing_results.png')
    plt.show()

    if len(original_image.shape) == 3:
        # RGB To BGR for cv2.imwrite saving images
        avg_filtered_bgr = cv2.cvtColor(avg_filtered, cv2.COLOR_RGB2BGR)
        median_filtered_bgr = cv2.cvtColor(median_filtered, cv2.COLOR_RGB2BGR)
        unsharp_masked_bgr = cv2.cvtColor(unsharp_masked, cv2.COLOR_RGB2BGR)

        cv2.imwrite('average_filter.png', avg_filtered_bgr)
        cv2.imwrite('median_filter.png', median_filtered_bgr)
        cv2.imwrite('unsharp_masking.png', unsharp_masked_bgr)
    else:
        cv2.imwrite('average_filter.png', avg_filtered)
        cv2.imwrite('median_filter.png', median_filtered)
        cv2.imwrite('unsharp_masking.png', unsharp_masked)

    print("Images saved complete!")

if __name__ == "__main__":
    main()