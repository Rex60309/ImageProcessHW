import numpy as np
import cv2
import matplotlib.pyplot as plt

def histogram_equalization_grayscale(img):
    """
    Apply histogram equalization to a grayscale image
    """
    # Calculate histogram
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])

    # Calculate cumulative distribution function (CDF)
    cdf = hist.cumsum()

    # Normalize CDF to range [0, 255]
    cdf_normalized = (cdf * 255 / cdf[-1]).astype('uint8')

    # Apply histogram equalization using the mapping
    img_equalized = cdf_normalized[img]

    return img_equalized

def histogram_equalization_color(img):
    """
    Apply histogram equalization to a color image using the method specified in the assignment
    """
    # Convert color image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization to grayscale image
    gray_equalized = histogram_equalization_grayscale(gray)

    # Create output color image
    img_equalized = np.zeros_like(img)

    # For each pixel, modify the color by (r,g,b) × G'/G
    # Handle division by zero
    ratio = np.zeros_like(gray, dtype=np.float32)
    mask = gray > 0
    ratio[mask] = gray_equalized[mask] / gray[mask]
    ratio[~mask] = 0

    # Apply the ratio to each color channel
    for i in range(3):
        img_equalized[:,:,i] = np.clip(img[:,:,i] * ratio, 0, 255).astype('uint8')

    return img_equalized, gray, gray_equalized

def plot_results(original, equalized, title_prefix):
    """
    Plot original and equalized images along with their histograms
    """
    plt.figure(figsize=(12, 8))

    # Original image
    plt.subplot(2, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title(f'{title_prefix} Original Image')
    plt.axis('off')

    # Original histogram
    plt.subplot(2, 2, 2)
    plt.hist(original.flatten(), 256, [0, 256], color='r')
    plt.title(f'{title_prefix} Original Histogram')
    plt.xlim([0, 256])

    # Equalized image
    plt.subplot(2, 2, 3)
    plt.imshow(equalized, cmap='gray')
    plt.title(f'{title_prefix} Equalized Image')
    plt.axis('off')

    # Equalized histogram
    plt.subplot(2, 2, 4)
    plt.hist(equalized.flatten(), 256, [0, 256], color='r')
    plt.title(f'{title_prefix} Equalized Histogram')
    plt.xlim([0, 256])

    plt.tight_layout()
    plt.savefig(f'{title_prefix}_results.png')
    plt.show()

def color_results(original, equalized, gray, gray_equalized):
    """
    Plot results for color image processing
    """
    # Plot grayscale conversion results
    plot_results(gray, gray_equalized, "Grayscale")

    # Plot color images
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Original Color Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(equalized, cv2.COLOR_BGR2RGB))
    plt.title('Equalized Color Image')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('color_results.png')
    plt.show()

    # Plot histograms for each channel of color images
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    titles = ['Blue', 'Green', 'Red']

    for i in range(3):
        # Original image channel histogram
        axs[0, i].hist(original[:,:,i].flatten(), 256, [0, 256], color=('b', 'g', 'r')[i])
        axs[0, i].set_title(f'Original {titles[i]} Channel Histogram')
        axs[0, i].set_xlim([0, 256])

        # Equalized image channel histogram
        axs[1, i].hist(equalized[:,:,i].flatten(), 256, [0, 256], color=('b', 'g', 'r')[i])
        axs[1, i].set_title(f'Equalized {titles[i]} Channel Histogram')
        axs[1, i].set_xlim([0, 256])

    plt.tight_layout()
    plt.savefig('color_histograms.png')
    plt.show()

def main():
    # Process grayscale image
    gray_image = cv2.imread('gray_image.png', cv2.IMREAD_GRAYSCALE)
    if gray_image is not None:
        equalized_gray = histogram_equalization_grayscale(gray_image)
        plot_results(gray_image, equalized_gray, "Grayscale")
        print("Grayscale image processing complete.")
    else:
        print("Failed to load grayscale image.")

    # Process color image
    color_image = cv2.imread('color_image.jpg')
    if color_image is not None:
        equalized_color, gray, gray_equalized = histogram_equalization_color(color_image)
        color_results(color_image, equalized_color, gray, gray_equalized)
        print("Color image processing complete.")
    else:
        print("Failed to load color image.")

    print("\nExperiment Discussion:")
    print("1. Grayscale Histogram Equalization:")
    print("   - The histogram equalization process redistributes pixel intensities to enhance contrast.")
    print("   - This is particularly effective for images with poor contrast or limited dynamic range.")
    print("   - The equalized histogram shows a more uniform distribution across all intensity levels.")

    print("\n2. Color Histogram Equalization:")
    print("   - The method preserves the color relationships while enhancing contrast.")
    print("   - By using the ratio of equalized grayscale to original grayscale, color balance is maintained.")
    print("   - Each color channel is modified proportionally based on the grayscale enhancement.")
    print("   - This approach prevents color distortion that might occur if equalizing each channel independently.")

if __name__ == "__main__":
    main()