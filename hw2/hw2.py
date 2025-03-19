import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from matplotlib.ticker import FuncFormatter
import os

# Apply histogram equalization to grayscale image
def histogram_equalization_grayscale(img):
    # Get image dimensions
    height, width = img.shape
    N = height * width
    L = 256  # (0-255)

    # Calculate histogram
    hist = np.zeros(L, dtype=int)
    for i in range(height):
        for j in range(width):
            hist[img[i, j]] += 1

    # Create dataframe for the table
    df = pd.DataFrame({
        'Gray level i': list(range(L)),
        'n_i': hist
    })
    df['∑n_i'] = df['n_i'].cumsum()
    df['∑(n_i/N)'] = df['∑n_i'] / N
    df['(L-1)∑(n_i/N)'] = df['∑(n_i/N)'] * (L-1)
    df['Rounded value'] = np.round(df['(L-1)∑(n_i/N)']).astype(int)

    # Apply mapping to the image
    img_equalized = np.zeros_like(img)
    for i in range(height):
        for j in range(width):
            img_equalized[i, j] = df['Rounded value'][img[i, j]]

    return img_equalized, df

# Apply histogram equalization to color image
def histogram_equalization_color(img):
    # Convert color image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_equalized, df = histogram_equalization_grayscale(gray)

    img_equalized = np.zeros_like(img, dtype=np.float32)

    # For each pixel, modify the color by (r,g,b) × G' / G
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            g_value = gray[i, j]
            g_prime = gray_equalized[i, j]

            # Avoid division by zero
            if g_value > 0:
                ratio = g_prime / g_value
            else:
                ratio = 0

            for c in range(3):
                img_equalized[i, j, c] = min(255, img[i, j, c] * ratio)

    return img_equalized.astype(np.uint8), gray, gray_equalized, df

# Plot original and equalized grayscale images with histograms
def plot_grayscale_results(original, equalized, df, filename_prefix):
    L = 256  # (0-255)

    hist_original = cv2.calcHist([original], [0], None, [L], [0, L])
    hist_equalized = cv2.calcHist([equalized], [0], None, [L], [0, L])

    plt.figure(figsize=(20, 12))

    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title(f'Original {filename_prefix}')
    plt.axis('off')

    # Original histogram
    plt.subplot(2, 3, 2)
    plt.bar(range(L), hist_original.flatten() / 1000, color='black', linewidth=1.25)
    plt.title(f'Original {filename_prefix} Histogram')
    plt.xlim([0, L-1])
    plt.ylim([0, np.max(hist_original.flatten() / 1000)])
    plt.gca().yaxis.set_major_formatter(FuncFormatter(k_formatter))

    # Original cumulative histogram
    cum_hist_original = np.cumsum(hist_original) / 1000
    plt.subplot(2, 3, 3)
    plt.plot(range(L), cum_hist_original, color='black', linewidth=1.25)
    plt.title(f'Original {filename_prefix} Cumulative Histogram')
    plt.xlim([0, L-1])
    plt.ylim([0, np.max(cum_hist_original)])
    plt.gca().yaxis.set_major_formatter(FuncFormatter(k_formatter))

    # Equalized image
    plt.subplot(2, 3, 4)
    plt.imshow(equalized, cmap='gray')
    plt.title(f'Equalized {filename_prefix}')
    plt.axis('off')

    # Equalized histogram
    plt.subplot(2, 3, 5)
    plt.bar(range(L), hist_equalized.flatten() / 1000, color='black', linewidth=1.25)
    plt.title(f'Equalized {filename_prefix} Histogram')
    plt.xlim([0, L-1])
    plt.ylim([0, np.max(hist_equalized.flatten() / 1000)])
    plt.gca().yaxis.set_major_formatter(FuncFormatter(k_formatter))

    # Equalized cumulative histogram
    cum_hist_equalized = np.cumsum(hist_equalized) / 1000
    plt.subplot(2, 3, 6)
    plt.plot(range(L), cum_hist_equalized, color='black', linewidth=1.25)
    plt.title(f'Equalized {filename_prefix} Cumulative Histogram')
    plt.xlim([0, L-1])
    plt.ylim([0, np.max(cum_hist_equalized)])
    plt.gca().yaxis.set_major_formatter(FuncFormatter(k_formatter))

    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_results.png')
    plt.show()

    # Save full table to CSV
    df.to_csv(f'{filename_prefix}_equalization_table.csv', index=False)

# Plot original and equalized color images with histograms
def plot_color_results(original, equalized, gray, gray_equalized, filename_prefix):
    plt.figure(figsize=(15, 10))

    # Original color image
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Original Color Image')
    plt.axis('off')

    # Original color to grayscale image
    plt.subplot(2, 2, 2)
    plt.imshow(gray, cmap='gray')
    plt.title('Original Color To Grayscale Image')
    plt.axis('off')

    # Equalized color image
    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(equalized, cv2.COLOR_BGR2RGB))
    plt.title('Equalized Color Image')
    plt.axis('off')

    # Equalized color to grayscale image
    plt.subplot(2, 2, 4)
    plt.imshow(gray_equalized, cmap='gray')
    plt.title('Equalized Color To Grayscale Image')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_results.png')
    plt.show()

    # Plot color channel histograms
    plt.figure(figsize=(20, 12))
    colors = ['b', 'g', 'r']
    titles = ['Blue', 'Green', 'Red']

    for i in range(3):
        hist = cv2.calcHist([original], [i], None, [256], [0, 256]).flatten() / 1000
        hist_eq = cv2.calcHist([equalized], [i], None, [256], [0, 256]).flatten() / 1000

        # Original histogram
        plt.subplot(4, 3, i + 1)
        plt.bar(range(256), hist, color=colors[i], linewidth=1.25, alpha=0.6)
        plt.title(f'Original {titles[i]} Channel Histogram')
        plt.xlim([0, 255])
        plt.ylim([0, np.max(hist)])
        plt.gca().yaxis.set_major_formatter(FuncFormatter(k_formatter))

        # Original wave chart
        plt.subplot(4, 3, i + 4)
        plt.plot(range(256), hist, color=colors[i], linewidth=1.5)
        plt.fill_between(range(256), hist, color=colors[i], alpha=0.4)
        plt.title(f'Original {titles[i]} Channel Wave Chart')
        plt.xlim([0, 255])
        plt.ylim([0, np.max(hist)])
        plt.gca().yaxis.set_major_formatter(FuncFormatter(k_formatter))

        # Equalized histogram
        plt.subplot(4, 3, i + 7)
        plt.bar(range(256), hist_eq, color=colors[i], linewidth=1.25, alpha=0.6)
        plt.title(f'Equalized {titles[i]} Channel Histogram')
        plt.xlim([0, 255])
        plt.ylim([0, np.max(hist_eq)])
        plt.gca().yaxis.set_major_formatter(FuncFormatter(k_formatter))

        # Equalized wave chart
        plt.subplot(4, 3, i + 10)
        plt.plot(range(256), hist_eq, color=colors[i], linewidth=1.5)
        plt.fill_between(range(256), hist_eq, color=colors[i], alpha=0.4)
        plt.title(f'Equalized {titles[i]} Channel Wave Chart')
        plt.xlim([0, 255])
        plt.ylim([0, np.max(hist_eq)])
        plt.gca().yaxis.set_major_formatter(FuncFormatter(k_formatter))

    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_color_channel_histograms.png')
    plt.show()

# Process images
def process_image(image_path_gray, image_path_color):
    img_gray = cv2.imread(image_path_gray, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print(f"Error: Could not read grayscale image {image_path_gray}")
        return

    img_color = cv2.imread(image_path_color)
    if img_color is None:
        print(f"Error: Could not read color image {image_path_color}")
        return

    # Grayscale image
    filename_prefix = 'Grayscale Image'
    equalized_gray, df = histogram_equalization_grayscale(img_gray)
    plot_grayscale_results(img_gray, equalized_gray, df, filename_prefix)

    # Color image
    equalized_color, gray, gray_equalized, df = histogram_equalization_color(img_color)
    filename_prefix = 'Color To Grayscale Image'
    plot_grayscale_results(gray, gray_equalized, df, filename_prefix)
    filename_prefix = 'Color Image'
    plot_color_results(img_color, equalized_color, gray, gray_equalized, filename_prefix)

# Unit display
def k_formatter(x, pos):
    return f'{int(x)}K'

# Input images
def main():
    image_path_gray = input("Enter grayscale image path: ").strip()
    image_path_color = input("Enter color image path: ").strip()

    missing_images = []
    if not os.path.exists(image_path_gray):
        missing_images.append("grayscale image")
    if not os.path.exists(image_path_color):
        missing_images.append("color image")

    if missing_images:
        print(f"Missing image: {', '.join(missing_images)}")
    else:
        process_image(image_path_gray, image_path_color)

if __name__ == "__main__":
    main()