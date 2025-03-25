import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# distinguish color or grayscale image and apply different filter
def average_or_median(image, kernel_size, a_or_m):
    # check grayscale or color image
    if len(image.shape) == 2:
        if a_or_m == "average":
            return apply_average_filter(image, kernel_size)
        else:
            return apply_median_filter(image, kernel_size)
    else:
        filtered_channels = []
        if a_or_m == "average":
            for channel in cv2.split(image):
                filtered_channel = apply_average_filter(channel, kernel_size)
                filtered_channels.append(filtered_channel)
        else:
            for channel in cv2.split(image):
                filtered_channel = apply_median_filter(channel, kernel_size)
                filtered_channels.append(filtered_channel)
        return cv2.merge(filtered_channels)

# average filter
def apply_average_filter(channel, kernel_size):
    m, n = channel.shape

    # half kernel size for padding calculation
    half_kernel = kernel_size // 2

    # average filter n*n mask
    mask = np.ones([kernel_size, kernel_size], dtype=float)
    mask /= (kernel_size * kernel_size)

    img_new = np.zeros([m, n], dtype=channel.dtype)

    for i in range(half_kernel, m-half_kernel):
        for j in range(half_kernel, n-half_kernel):
            # extract the local region
            region = channel[i-half_kernel:i+half_kernel+1, j-half_kernel:j+half_kernel+1]
            # apply the mask
            temp = np.sum(region * mask)
            img_new[i, j] = temp

    return img_new

# median filter
def apply_median_filter(channel, kernel_size):
    m, n = channel.shape

    half_kernel = kernel_size // 2

    img_new = np.zeros([m, n], dtype=channel.dtype)

    for i in range(half_kernel, m-half_kernel):
        for j in range(half_kernel, n-half_kernel):
            # extract the local region
            temp = channel[i-half_kernel:i+half_kernel+1, j-half_kernel:j+half_kernel+1].flatten()
            # sort and get median
            temp_sorted = sorted(temp)
            median_index = len(temp_sorted) // 2
            img_new[i, j] = temp_sorted[median_index]

    return img_new

# unsharp masking
def unsharp_masking(image, kernel_size, k):
    # blur image
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    # create mask (original image - blurred image)
    mask = cv2.subtract(image.astype(float), blurred.astype(float))

    # scale the mask
    scaled_mask = k * mask

    # add the scaled mask to original image
    sharpened = cv2.add(image.astype(float), scaled_mask)

    # clip values to ensure they are within the valid range
    sharpened = np.clip(sharpened, 0, 255).astype(image.dtype)

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

    # input kernel size
    kernel_size = int(input("Enter kernel size for n by n (odd number): "))
    if kernel_size % 2 == 0:
        kernel_size += 1

    # average filter
    avg_filtered = average_or_median(original_image_rgb, kernel_size, "average")

    # median filter
    median_filtered = average_or_median(original_image_rgb, kernel_size, "median")

    k = int(input("Enter scale k for unsharp masking: "))
    # unsharp masking
    unsharp_masked = unsharp_masking(original_image_rgb, kernel_size, k)

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

    # check if color image
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