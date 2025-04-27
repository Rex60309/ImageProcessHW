import numpy as np
import matplotlib.pyplot as plt

# Part 1: Create an image g(x,y) with all pixels having gray value of 100
def create_constant_image(size=(256, 256), gray_value=100):
    return np.ones(size, dtype=np.uint8) * gray_value

# Part 2: Generate Gaussian noise with mean=0, variance=25
def generate_gaussian_noise(size=(256, 256), mean=0, variance=25):
    sigma = np.sqrt(variance)
    return np.random.normal(mean, sigma, size).astype(np.int16)

# Generate noisy image f(x,y) = g(x,y) + n(x,y)
def add_noise_to_image(image, noise):
    noisy_image = image.astype(np.int16) + noise
    # Clip values to valid range [0, 255]
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

# Part 3: Display histogram
def plot_histogram(image, title='Histogram'):
    hist = np.zeros(256)
    for i in range(256):
        hist[i] = np.sum(image == i)

    plt.figure(figsize=(10, 4))
    plt.bar(range(256), hist)
    plt.title(title)
    plt.xlabel('Gray Level')
    plt.ylabel('Frequency')
    plt.xlim([0, 255])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return hist

def main():
    # Part 1: Create constant gray value image
    image_size = (256, 256)
    g_xy = create_constant_image(image_size)

    # Part 2: Generate Gaussian noise and add to image
    noise = generate_gaussian_noise(image_size)
    f_xy = add_noise_to_image(g_xy, noise)

    # Display results
    plt.figure(figsize=(12, 10))

    # Display original constant image
    plt.subplot(2, 2, 1)
    plt.imshow(g_xy, cmap='gray', vmin=0, vmax=255)
    plt.title('Original Image g(x,y) with Gray Value = 100')
    plt.colorbar()

    # Display noisy image
    plt.subplot(2, 2, 2)
    plt.imshow(f_xy, cmap='gray', vmin=0, vmax=255)
    plt.title('Noisy Image f(x,y) = g(x,y) + n(x,y)')
    plt.colorbar()

    # Display histogram of original image
    plt.subplot(2, 2, 3)
    hist_g = plot_histogram(g_xy, 'Histogram of g(x,y)')
    plt.tight_layout()

    # Display histogram of noisy image
    plt.subplot(2, 2, 4)
    hist_f = plot_histogram(f_xy, 'Histogram of f(x,y)')
    plt.tight_layout()

    plt.savefig('assignment_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Part 4: Comment on results
    print("\nComments on Results:")
    print("1. The original image has a constant gray value of 100, so its histogram shows a single spike at 100.")
    print("2. After adding Gaussian noise with mean=0 and variance=25, the histogram spreads out in a bell curve shape.")
    print("3. The noisy image's histogram is centered around 100 (original value) with spread determined by variance=25.")
    print("4. The standard deviation is 5 (sqrt of variance 25), so most values fall within 100±15 (3 standard deviations).")
    print("5. The clipping at 0 and 255 might affect the perfect Gaussian shape at the extremes.")

if __name__ == "__main__":
    main()